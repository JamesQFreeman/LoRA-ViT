# Sheng Wang at Feb 22 2023

import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from base_vit import ViT


class LoRA_ViT(nn.Module):
    """Some Information about LoRA_ViT"""

    def __init__(self, vit_model: ViT, dim: int, r: int):
        super(LoRA_ViT, self).__init__()

        assert r > 0

        # create for storage
        self.w_As = []
        self.w_Bs = []

        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # get wq and wv in our vit
        for blk in vit_model.transformer.blocks:
            w_q_linear = blk.attn.proj_q
            w_v_linear = blk.attn.proj_v
            assert dim == w_q_linear.in_features

            w_A = nn.parameter.Parameter(torch.Tensor(r, dim))
            w_B = nn.parameter.Parameter(torch.Tensor(dim, r))
            self.w_As.append(w_A)
            self.w_Bs.append(w_B)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            # 5 is a magic number from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py line 125
            nn.init.kaiming_uniform_(w_A, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B)

    def forward(self, x: Tensor) -> Tensor:

        return x


model = ViT('B_16_imagenet1k')
LoRA(model, 4)
