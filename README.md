# LoRA-ViT
Low rank adaptation for Vision Transformer

## Installation
Gii clone. My ```torch.__version__==1.13.0```, other version newer than ```torch.__version__==1.10.0``` should also work, I guess.
You may also need a ```safetensors``` from huggingface to load and save weight.

## Usage
Wrap you ViT using LoRA-ViT
```python
from base_vit import ViT
import torch
from lora import LoRA_ViT

model = ViT('B_16_imagenet1k')
model.load_state_dict(torch.load('B_16_imagenet1k.pth'))
preds = model(img) # preds.shape = torch.Size([1, 1000])

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable parameters: {num_params}") #trainable parameters: 86859496


lora_model = LoRA_ViT(model, r=4)
num_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
print(f"trainable parameters: {num_params}") # trainable parameters: 147456

```
Save and load LoRA:
```python
lora_model.save_lora_parameters('mytask.lora.safetensors') # save
lora_model.load_lora_parameters('mytask.lora.safetensors') # load
```
## Credit
ViT code and imagenet pretrained weight come from ```lukemelas/PyTorch-Pretrained-ViT```