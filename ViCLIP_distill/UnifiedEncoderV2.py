import torch
import torch.nn as nn
from einops import rearrange

from UnifiedEncoderBase import UnifiedEncoderBase
from typing import override

class UnifiedEncoderV2(UnifiedEncoderBase):
    def __init__(self, config, max_txt_l, tokenizer):
        
        super().__init__(config, max_txt_l, tokenizer)
    
    @override
    def forward(self, image, text, *args, **kwargs):
        ...
    
    @override
    def forward_image(self, image):
        ...
    
    @override
    def forward_text(self, text):
        ...