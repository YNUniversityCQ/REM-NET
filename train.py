from __future__ import print_function, division
import os
import time
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from models import modules
import torch.optim as optim
from dataloaders import loaders
from torchsummary import summary
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import warnings

if __name__ == '__main__':
    print('GPU:', torch.cuda.device_count())

    device = torch.device("cuda:0")

    warnings.filterwarnings("ignore")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    Radio_train = loaders.loader_3D(phase="train")
    Radio_val = loaders.loader_3D(phase="val")
    Radio_test = loaders.loader_3D(phase="test")

    image_datasets = {
        'train': Radio_train, 'val': Radio_val
    }

    batch_size = 2

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    }

    # 加载网络参数
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.backends.cudnn.enabled
    model =modules.REM_Net()
    model.cuda()
    # summary(model, input_size=[(1, 256, 256),
    #                            (1, 256, 256)])

    def calc_loss_dense(pred, target, metrics):
        criterion = nn.MSELoss()
        loss = criterion(pred, target)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss

    def calc_loss_sparse(pred, target, samples, metrics, num_samples):
        criterion = nn.MSELoss()
        loss = criterion(samples*pred, samples*target)*(256**2)/num_samples
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss

    def print_metrics(metrics, epoch_samples, phase):
        outputs1 = []
        outputs2 = []
        for k in metrics.keys():
            outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

        print("{}: {}".format(phase, ", ".join(outputs1)))

    def train_model(model, optimizer, scheduler, num_epochs=35, targetType="dense"):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            since = time.time()

            # 训练并验证
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("learning rate", param_group['lr'])

                    model.train()
                else:
                    model.eval()

                metrics = defaultdict(float)
                epoch_samples = 0
                if targetType == "dense":

                    # # if using path loss map
                    # for build, antenna, sample, target, name in tqdm(dataloaders[phase]):
                    # if not using path loss map
                    for build, antenna, target, name in tqdm(dataloaders[phase]):

                        # # if using path loss map
                        # build, antenna, sample, target = build.to(device), antenna.to(device), sample.to(device), target.to(device)
                        # if not using path loss map
                        build, antenna, target = build.to(device), antenna.to(device),  target.to(device)

                        # 梯度归零
                        optimizer.zero_grad()

                        # 前向传播
                        with torch.set_grad_enabled(phase == 'train'):

                            # if not using path loss map
                            # outputs = model(build, antenna, sample)

                            # if not using path loss map
                            outputs = model(build, antenna)

                            loss = calc_loss_dense(outputs, target, metrics)

                            # 反向传播梯度更新
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                        # 统计数据
                        epoch_samples += target.size(0)

                print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['loss'] / epoch_samples

                # 复制模型参数并保存模型
                if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Best val loss: {:4f}'.format(best_loss))

        # 加载最优权重
        model.load_state_dict(best_model_wts)
        return model

    # 执行训练
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    model = train_model(model, optimizer_ft, exp_lr_scheduler)

    # 创建模型保存文件
    try:
        os.mkdir('model_result')
    except OSError as error:
        print(error)

    # 保存第一个模型
    torch.save(model.state_dict(), 'model_result/Best_Weight.pt')
