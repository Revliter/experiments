import torch
import torch.nn as nn
from einops import rearrange

from .UnifiedEncoderBase import UnifiedEncoderBase

class UnifiedEncoderV2(UnifiedEncoderBase):
    """
    UnifiedEncoder V2:
        [] Use the structure in ViLT
        [] Follow CoCa's method:
            [] Use different queries for different tasks.
            [] Use cross attention for bootstrapping the modality-specific embeddings.
    """

    def __init__(self, config, max_txt_l, tokenizer):

        super().__init__(config, max_txt_l, tokenizer, use_multimodality_queries=True)
        
    
    def forward_image(self, x, masking_prob=0.0, return_embed=False) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.image_positional_embedding.to(x.dtype)
        x = x + self.token_type_embeddings[0].to(x.dtype)
        
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
        x = self.transformer(x, modality_type=0)

        x = self.image_post_norm(x)

        if return_embed:
            return x.permute(1, 0, 2)# @ self.proj, self.dropout(x[0]) @ self.proj
        else:
            if self.image_projection is not None:
                x = self.dropout(x[0]) @ self.image_projection
            else:
                x = x.permute(1, 0, 2)  #NBD -> BND
            return x
    
    def forward_text(self, text, return_embed=False) -> torch.Tensor:
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.text_positional_embedding.to(x.dtype)
        x = x + self.token_type_embeddings[1].to(x.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, modality_type=1)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.text_post_norm(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        if return_embed:
            return x @ self.text_projection, x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        else:
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x