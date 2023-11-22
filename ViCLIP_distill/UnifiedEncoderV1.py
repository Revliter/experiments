import torch
import torch.nn as nn
from einops import rearrange

from UnifiedEncoderBase import UnifiedEncoderBase
from typing import override

class UnifiedEncoderV1(UnifiedEncoderBase):
    """
    UnifiedEncoder V1:
        [] Use the structure in ViLT
    """

    def __init__(self, config, max_txt_l, tokenizer):

        super().__init__(config, max_txt_l, tokenizer)

    @override
    def forward(
        self, 
        image, 
        text, 
        raw_text, 
        idx,
        log_generation=None, 
        return_sims: bool = False,
        input_text_only: bool = False,
        input_video_only: bool = False
    ):
        '''
        For image and text, we compute the embedding separately, since they are pairs, we add the two embeddings together.
        '''
        assert not input_text_only and not input_video_only
        
        # preprocess image
        if not input_text_only:
            if image.ndim == 5:
                image = image.permute(0, 2, 1, 3, 4).contiguous()
            else:
                image = image.unsqueeze(2)
            image_embeds = self.forward_image(image, masking_prob=self.model.masking_prob)
        else:
            image_embeds = 0
        
        # preprocess text
        if not input_video_only:
            text = self.tokenize(text, context_length=self.max_txt_l)
            text_embeds = self.forward_text(text)
        else:
            text_embeds = 0
        
        final_embed = image_embeds + text_embeds
        
        return final_embed
        
    @override
    def forward_image(self, x, masking_prob=0.0, return_embed=False) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = x + self.token_type_embeddings(torch.full_like(x))
        
        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        if hasattr(self, 'temporal_positional_embedding'):
            if x.size(1) == 1:
                # This is a workaround for unused parameter issue
                x = x + self.temporal_positional_embedding.mean(1)
            else:
                x = x + self.temporal_positional_embedding
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)

        if masking_prob > 0.0:
            x = self.mask_tokens(x, masking_prob) # 这里mask是rand mask，所以后面无法再区分各帧特征了

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.image_pre_norm(x)

        x = x.permute(1, 0, 2)  #BND -> NBD
        x = self.transformer(x)

        x = self.image_post_norm(x)

        if return_embed:
            return x.permute(1, 0, 2)# @ self.proj, self.dropout(x[0]) @ self.proj
        else:
            if self.proj is not None:
                x = self.dropout(x[0]) @ self.image_projection
            else:
                x = x.permute(1, 0, 2)  #NBD -> BND
            return x
    
    @override
    def forward_text(self, text, return_embed=False) -> torch.Tensor:
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x + self.token_type_embeddings(torch.zeros_like(x))
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.text_post_norm(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if return_embed:
            return x @ self.text_projection, x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        else:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x