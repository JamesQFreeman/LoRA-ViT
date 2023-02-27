# Sheng Wang at Feb 22 2023

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import save_file
from base_vit import ViT
from torch.nn.parameter import Parameter


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
    """Some Information about LoRA_ViT
        Examples::
            >>> model = ViT('B_16_imagenet1k')
            >>> lora_model = LoRA_ViT(model, dim=768, r=4)
            >>> preds = lora_model(img)
            >>> print(preds.shape)
            torch.Size([1, 1000])
    """

    def __init__(self,
                 vit_model: ViT,
                 dim: int,
                 r: int,
                 num_classes: int = 0):
        super(LoRA_ViT, self).__init__()

        assert r > 0

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
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
            blk.attn.proj_q = _LoRALayer(
                w_q_linear, w_a_linear_q, w_b_linear_q)
            blk.attn.proj_v = _LoRALayer(
                w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.fc = nn.Linear(
                vit_model.fc.out_features, num_classes)

    def save_lora_parameters(self,
                             filename: str) -> None:
        r"""Only safetensors is supported now.

          pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith('.safetensors')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {
            f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)
        }
        b_tensors = {
            f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)
        }
        merged_dict = {**a_tensors, **b_tensors}
        save_file(merged_dict, filename)

    def load_lora_parameters(self,
                             filename: str) -> None:
        r"""Only safetensors is supported now.

          pip install safetensor if you do not have one installed yet.
        """

        assert filename.endswith('.safetensors')

        with safe_open(filename, framework="pt") as f:
            for i, w_A_linear in enumerate(self.w_As):
                saved_key = f"w_a_{i:03d}"
                w_A_linear.weight = Parameter(f.get_tensor(saved_key))

            for i, w_B_linear in enumerate(self.w_Bs):
                saved_key = f"w_b_{i:03d}"
                w_B_linear.weight = Parameter(f.get_tensor(saved_key))

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)
