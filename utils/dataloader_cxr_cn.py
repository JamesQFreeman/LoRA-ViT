import os

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
        for case in os.listdir(f"ChinaSet_AllFiles/CXR_png"):
            if ".png" in case:
                cases = cases + [f"ChinaSet_AllFiles/CXR_png/{case}"]
        cases.sort()

        train_index, test_index = list(kf.split(cases))[fold_idx]
        if data_type == "train":
            idx = train_index
        elif data_type == "test":
            idx = test_index
        else:
            print("Dataset type error")
            exit()
        self.cases = np.array(cases)[idx]

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
        
        label_id = self.cases[idx].split("/")[2].split(".")[0]
        with open("ChinaSet_AllFiles/ClinicalReadings/" + label_id + ".txt", "r") as f:
            data = f.readlines()
            if data[1] == "normal":
                label = 0
            else:
                label = 1
        label = torch.tensor(label, dtype=torch.long)
        return image, label


def cxrDataloader(cfg):
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

