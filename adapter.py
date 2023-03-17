import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from safetensors import safe_open
from safetensors.torch import save_file
from base_vit import ViT
from torch.nn.parameter import Parameter
from timm.models.vision_transformer import VisionTransformer as timm_ViT
import timm

class Adapter_ViT(nn.Module):
    """Applies mlp adapter to a vision transformer.

    Args:
        vit_model: a vision transformer model, see base_vit.py
        num_layers: number of hidden layers
        num_classes: how many classes the model output, default to the vit model

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> adapter_model = Adapter_ViT(model, r=4)
        >>> preds = adapter_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """
    
    def __init__(self,
                vit_model: timm_ViT,
                num_classes: int = 0):
        super(Adapter_ViT, self).__init__()
        
        assert num_classes > 0
        
        for param in vit_model.parameters():
            param.requires_grad = False
        
        self.dim = vit_model.blocks[0].attn.qkv.in_features
        self.adapter = nn.Sequential()
        for t_layer_i in range(len(vit_model.blocks)//2):
            self.adapter.add_module("layer_" + str(t_layer_i), nn.Linear(self.dim, self.dim))
            self.adapter.add_module("relu_" + str(t_layer_i), nn.ReLU())
        self.adapter.add_module("fc", nn.Linear(self.dim, num_classes))
        
        self.backbone = vit_model
        self.backbone.head = self.adapter
        
    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)
    
    
# import timm
# from transformers import AutoFeatureExtractor, AutoModelForImageClassification
# from lora import LoRA_ViT_timm
# hiddenSize={
#             "small":768,
#             "base":768,
#             "large":1024,
#             "huge":1280
#             }
# weightInfo={
#             # "small":"WinKawaks/vit-small-patch16-224",
#             "base":"vit_base_patch16_224.orig_in21k_ft_in1k",
#             "base_dino":"vit_base_patch16_224.dino", # 21k -> 1k
#             "base_sam":"vit_base_patch16_224.sam", # 1k
#             "base_mill":"vit_base_patch16_224_miil.in21k_ft_in1k", # 1k
#             "base_beit":"beitv2_base_patch16_224.in1k_ft_in22k_in1k",
#             "base_clip":"vit_base_patch16_clip_224.laion2b_ft_in1k", # 1k
#             "base_deit":"deit_base_distilled_patch16_224", # 1k
#             # "large":"google/vit-large-patch16-224",
#             "large_clip":"vit_large_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
#             "large_beit":"beitv2_large_patch16_224.in1k_ft_in22k_in1k", 
#             "huge_clip":"vit_huge_patch14_clip_224.laion2b_ft_in1k", # laion-> 1k
#             # "giant_eva":"eva_giant_patch14_224.clip_ft_in1k", # laion-> 1k
#             # "giant_clip":"vit_giant_patch14_clip_224.laion2b",
#             # "giga_clip":"vit_gigantic_patch14_clip_224.laion2b"
#             }
# for i in list(weightInfo):
#     print(f"========={i}=========")
#     model = timm.create_model(weightInfo[i], pretrained=True)
#     lora_model = LoRA_ViT_timm(model, r=4, dim=hiddenSize[i.split('_')[0]], num_classes=14)