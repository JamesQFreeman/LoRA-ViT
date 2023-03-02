import os
import csv
import random

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize
from torchvision.transforms import Normalize


class GraphDataset(Dataset):
    def __init__(self, data_type="train", fold_idx=0):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        cases = []
        if data_type == "train":
            for grade in os.listdir(f"blood-cells/dataset2-master/dataset2-master/images/TRAIN"):
                _cases = os.listdir(f"blood-cells/dataset2-master/dataset2-master/images/TRAIN/{grade}")
                _cases = [f"blood-cells/dataset2-master/dataset2-master/images/TRAIN/{grade}/{_}" for _ in _cases if ".jpeg" in _]
                cases = cases + _cases
        elif data_type == "test":
            for grade in os.listdir(f"blood-cells/dataset2-master/dataset2-master/images/TEST"):
                _cases = os.listdir(f"blood-cells/dataset2-master/dataset2-master/images/TEST/{grade}")
                _cases = [f"blood-cells/dataset2-master/dataset2-master/images/TEST/{grade}/{_}" for _ in _cases if ".jpeg" in _]
                cases = cases + _cases
        else:
            print("Dataset type error")
            exit()

        random.shuffle(cases)
        self.cases = cases

    def __len__(self):
        # return 100
        return len(self.cases)

    def __getitem__(self, idx):
        resize = Resize([384, 384])
        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image = np.array(Image.open(self.cases[idx]).convert("RGB")).astype(np.float32) / 255.0
        image = rearrange(torch.tensor(image, dtype=torch.float32), 'h w c -> c h w')
        image = resize(image)
        image = normalize(image)
        
        label_path = str(self.cases[idx].split("/")[5])
        # NEUTROPHIL 0 MONOCYTE 1 EOSINOPHIL 2 LYMPHOCYTE 3
        if label_path == "NEUTROPHIL":
            label = 0
        elif label_path == "MONOCYTE":
            label = 1
        elif label_path == "EOSINOPHIL":
            label = 2
        elif label_path == "LYMPHOCYTE":
            label = 3
        label = np.array(label)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


def BloodDataloader(cfg):
    train_set = DataLoader(
        GraphDataset(data_type="train", fold_idx=cfg.fold),
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    test_set = DataLoader(
        GraphDataset(data_type="test", fold_idx=cfg.fold),
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    return train_set, test_set

