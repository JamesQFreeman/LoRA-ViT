import time
import torch
from base_vit import ViT
from lora import LoRA_ViT
import numpy as np


BATCH_SIZE = 1
GPU = False

img = torch.randn(BATCH_SIZE, 3, 384, 384)
target = torch.randn(BATCH_SIZE, 1000)
criterion = torch.nn.MSELoss()


class TimeProfile:
    @staticmethod
    def test_base():
        model = ViT('B_16_imagenet1k')
        preds = model(img)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = criterion(preds, target)
        start_time = time.time()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        time_cost = end_time - start_time
        print(f"base backpropagation took {time_cost:.4f} seconds")
        return time_cost

    @staticmethod
    def test_lora():
        model = ViT('B_16_imagenet1k')
        lora_model = LoRA_ViT(model, r=4)
        preds = lora_model(img)
        optimizer = torch.optim.SGD(lora_model.parameters(), lr=0.1)
        loss = criterion(preds, target)
        start_time = time.time()
        loss.backward()
        optimizer.step()
        end_time = time.time()
        time_cost = end_time - start_time
        print(f"LoRA backpropagation took {time_cost:.4f} seconds")
        return time_cost


class GRAMProfile:
    pass


results_base = np.array([TimeProfile.test_base() for _ in range(10)])
results_lora = np.array([TimeProfile.test_lora() for _ in range(10)])
print(f"Base\nMean:{results_base.mean()} Std:{results_base.std()}")
print(f"LoRA\nMean:{results_lora.mean()} Std:{results_lora.std()}")
