import torch
from torch import nn
from einops import rearrange
from pkg_resources import packaging

from collections import OrderedDict
from timm.models.layers import DropPath

import torch.utils.checkpoint as checkpoint

import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text



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
    
    def __init__(self, config, max_txt_l, tokenizer, drop_path=0.1, checkpoint_num=0, dropout=0., temp_embed=True):
        super(UnifiedEncoderBase, self).__init__()
        self.config = config
        
        self._tokenizer = SimpleTokenizer()
        self.max_txt_l = max_txt_l
    
        scale = config.width ** -0.5
        self.token_type_embeddings = nn.Parameter(scale * torch.randn(2, config.width))
        self.class_embedding = nn.Parameter(scale * torch.randn(config.width))
        self.image_positional_embedding = nn.Parameter(scale * torch.randn((config.input_resolution // config.patch_size) ** 2 + 1, config.width))
        self.text_positional_embedding = nn.Parameter(scale * torch.randn(config.context_length, config.width))
        
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
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        
        self.image_pre_norm = LayerNorm(config.width)
        self.image_post_norm = LayerNorm(config.width)
        self.text_post_norm = LayerNorm(config.width)
        
        self.text_projection = nn.Parameter(torch.empty(config.width, config.output_dim))
        self.text_projection.data.normal_(mean=0.0, std=0.02)
        self.image_projection = nn.Parameter(torch.empty(config.width, config.output_dim))
        self.image_projection.data.normal_(mean=0.0, std=0.02)
    
    def forward(
        self, 
        image, 
        text,
        input_text_only: bool = False,
        input_video_only: bool = False
    ):
        '''
        For image and text, we compute the embedding separately, since they are pairs, we add the two embeddings together.
        '''
        assert not input_text_only and not input_video_only
        
        # preprocess image
        if not input_text_only:
            image_embeds = self.encode_vision(image)
        else:
            image_embeds = 0
        
        # preprocess text
        if not input_video_only:
            text_embeds = self.encode_text(text)
        else:
            text_embeds = 0
        
        return image_embeds, text_embeds
    
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
        device = next(self.transformer.parameters()).device
        text = self.tokenize(text, context_length=self.max_txt_l).to(device)
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
    
    def mask_tokens(self, inputs, masking_prob=0.0):
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