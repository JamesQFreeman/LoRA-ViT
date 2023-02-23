# Sheng Wang at Feb 22 2023

import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from base_vit import ViT


class _LoRALayer(nn.Module):
    def __init__(self,
                 w: nn.Module,
                 w_a: nn.Module,
                 w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x)+self.w_b(self.w_a(x))
        return x


class LoRA_ViT(nn.Module):
    """Some Information about LoRA_ViT"""

    def __init__(self,
                 vit_model: ViT,
                 dim: int,
                 r: int):
        super(LoRA_ViT, self).__init__()

        assert r > 0

        # create for storage, then we can init them or load weights
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
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.lora_vit = vit_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)


# model = ViT('B_16_imagenet1k')
# lora_vit = LoRA_ViT(model, 768, 4)
# img = torch.randn(1, 3, 384, 384)
# preds = lora_vit(img)
# print(preds.shape)
