import torch
import os
import cv2
from torchmetrics import R2Score
import numpy as np
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataloaders import loaders
from models import modules
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

device = torch.device("cuda:0")
model = modules.DeepAE()
model.load_state_dict(torch.load('model_result/Deep_AE.pt'))
model.to(device)

def my_loss(output, target):
    loss = torch.sum((output - target) ** 2, 1)
    loss = torch.sqrt(loss)
    loss = torch.mean(loss)
    return loss

def calc_loss_dense(pred, target):
    loss = my_loss(pred, target)
    return loss

def main_worker():

    # 加载测试数据
    test_data = loaders.loader_3D(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)

    interation = 0
    loss = []
    for sample, mask, target, img_name in tqdm(test_dataloader):
        interation += 1

        sample, mask, target = sample.cuda(), mask.cuda(), target.cuda()

        with torch.no_grad():
            pre = model(sample, mask)
            losses = calc_loss_dense(pre, target)

        loss.append(losses)

    rmse_err = sum(loss)/len(loss)

    print('测试集均方根误差：', rmse_err)

if __name__ == '__main__':
 main_worker()