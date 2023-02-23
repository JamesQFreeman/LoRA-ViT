# LoRA-ViT
 Low rank adaptation for Vision Transformer

## Installation
Gii clone. My ```torch.__version__==1.8.0```, other version should also work, I guess.

## Usage
```python
from base_vit import ViT
import torch
from lora import LoRA_ViT

model = ViT('B_16_imagenet1k')
model.load_state_dict(torch.load('B_16_imagenet1k.pth'))
preds = model(img) # preds.shape = torch.Size([1, 1000])

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable parameters: {num_params}") #trainable parameters: 86859496


lora_model = LoRA_ViT(model, dim=768, r=4)
num_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"trainable parameters: {num_params}") # trainable parameters: 147456

```
## Credit
ViT code comes form ```lukemelas/PyTorch-Pretrained-ViT```