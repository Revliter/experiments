import torch
from torch import nn
from einops import rearrange
from typing import override
from pkg_resources import packaging

from collections import OrderedDict
from timm.models.layers import DropPath

import torch.utils.checkpoint as checkpoint

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, drop_path=0., attn_mask=None, dropout=0.):
        super().__init__()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # logger.info(f'Droppath: {drop_path}')
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("drop1", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("drop2", nn.Dropout(dropout)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.drop_path1(self.attention(self.ln_1(x)))
        x = x + self.drop_path2(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, drop_path=0., checkpoint_num=0, dropout=0.):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]
        self.resblocks = nn.ModuleList()
        for idx in range(layers):
            self.resblocks.append(ResidualAttentionBlock(width, heads, drop_path=dpr[idx], dropout=dropout))
        self.checkpoint_num = checkpoint_num

    def forward(self, x):
        for idx, blk in enumerate(self.resblocks):
            if idx < self.checkpoint_num:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class UnifiedEncoderBase(nn.Module):
    """
    A unified encoder for both image and text
    """
    
    def __init__(self, config, max_txt_l, tokenizer, vocab_size=0, transformer_width=0, drop_path=0.1, checkpoint_num=0, dropout=0., temp_embed=True):
        self._tokenizer = tokenizer
        self.max_txt_l = max_txt_l
        
        self.token_type_embeddings = nn.Embedding(2, config.width)
        self.token_type_embeddings.apply(init_weights)
    
        scale = config.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(config.width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((config.input_resolution // config.patch_size) ** 2 + 1, config.width))
    
        if temp_embed:
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(1, config.num_frames, config.width))
    
        self.transformer = Transformer(
            config.width, config.layers, config.heads, drop_path=drop_path, checkpoint_num=checkpoint_num,
            dropout=dropout
        )
        
        self.conv1 = nn.Conv3d(
            3, config.width, 
            (config.kernel_size, config.patch_size, config.patch_size), 
            (config.kernel_size, config.patch_size, config.patch_size), 
            (0, 0, 0), bias=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        
        self.image_pre_norm = LayerNorm(transformer_width)
        self.image_post_norm = LayerNorm(transformer_width)
        self.text_post_norm = LayerNorm(transformer_width)
        
        self.text_projection = nn.Parameter(torch.empty(config.width, config.output_dim))
        self.text_projection.data.normal_(mean=0.0, std=0.02)
        self.image_projection = nn.Parameter(torch.empty(config.width, config.output_dim))
        self.image_projection.data.normal_(mean=0.0, std=0.02)
    
    def encode_vision(self, image, test=False) -> torch.Tensor:
        if image.ndim == 5:
            image = image.permute(0, 2, 1, 3, 4).contiguous()
        else:
            image = image.unsqueeze(2)
        if not test and self.config.masking_prob > 0.0:
            return self.forward_image(
                image, masking_prob=self.config.masking_prob
            )

        return self.forward_image(image)
    
    def encode_text(self, text) -> torch.Tensor:
        text = self.tokenize(text, context_length=self.max_txt_l)
        text_embeds = self.forward_text(text)
        return text_embeds
    
    def forward_image(self, image, masking_prob=0.0, return_embed=False) -> torch.Tensor:
        ...
    
    def forward_text(self, text) -> torch.Tensor:
        ...
    
    def tokenize(self, texts, context_length=77, truncate=True):
        """
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self._tokenizer.encode(text) + [eot_token] for text in texts]
        if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
        else:
            result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result
    
    def mask_temporal_tokens(self, inputs, masking_prob=0.0):
        B, L, _ = inputs.shape

        # This is different from text as we are masking a fix number of tokens
        Lm = int(masking_prob * L)
        masked_indices = torch.zeros(B, L)
        indices = torch.argsort(torch.rand_like(masked_indices), dim=-1)[:, :Lm]
        batch_indices = (
            torch.arange(masked_indices.shape[0]).unsqueeze(-1).expand_as(indices)
        )
        masked_indices[batch_indices, indices] = 1

        masked_indices = masked_indices.bool()

        return inputs[~masked_indices].reshape(B, -1, inputs.shape[-1])