import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from base_vit import ViT
from lora import LoRA_ViT
from utils.dataloader import kneeDataloader
from utils.result import ResultCLS as Result
from utils.utils import init, save


def train(epoch):
    running_loss = 0.0
    this_lr = scheduler.get_last_lr()[0]
    net.train()
    for image, label in tqdm(trainset, ncols=60, desc="train", unit="b", leave=None):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        with autocast(enabled=True):
            pred = net.forward(image)
            loss = loss_func(pred, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss = running_loss + loss.item()
    scheduler.step()

    loss = running_loss / len(trainset)
    logging.info(f"\n\nEPOCH: {epoch}, LOSS : {loss:.3f}, LR: {this_lr:.2e}")
    return


@torch.no_grad()
def eval(epoch):
    result.init()
    net.eval()
    for image, label in tqdm(testset, ncols=60, desc='test', unit="b", leave=None):
        image, label = image.to(device), label.to(device)
        with autocast(enabled=True):
            pred = net.forward(image)
            result.eval(label, pred)
    result.print(epoch)
    return


if __name__ == "__main__":
    scaler = GradScaler()
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", type=int, default=16)
    parser.add_argument("-fold", type=int, default=0)
    parser.add_argument("-lr", type=float, default=3e-4)
    parser.add_argument("-num_workers", type=int, default=4)
    cfg = parser.parse_args()
    ckpt_path = init()
    device = 'cuda'

    model = ViT('B_16_imagenet1k')
    model.load_state_dict(torch.load('B_16_imagenet1k.pth'))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {num_params/2**20:.1f}M") #trainable parameters: 86859496

    lora_model = LoRA_ViT(model, dim=768, r=4)
    num_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    print(f"trainable parameters: {num_params/2**20:.1f}M") # trainable parameters: 147456

    net = lora_model.to(device)
    trainset, testset = kneeDataloader(cfg)

    loss_func = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, 100, 1e-6)
    result = Result()

    for epoch in range(1, 101):
        train(epoch)
        eval(epoch)
        save(result, net, ckpt_path)

