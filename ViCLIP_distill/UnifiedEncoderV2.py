import torch
import torch.nn as nn
from einops import rearrange

from UnifiedEncoderBase import UnifiedEncoderBase

class UnifiedEncoderV2(UnifiedEncoderBase):
    def __init__(self, config, max_txt_l, tokenizer):
        
        super().__init__(config, max_txt_l, tokenizer)

    def forward(self, image, text, *args, **kwargs):
        ...

    def forward_image(self, image):
        ...
    
    def forward_text(self, text):
        ...