from __future__ import print_function, division
import os
import torch
import warnings
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models

warnings.filterwarnings("ignore")

class loader_3D(Dataset):

    def __init__(self, maps=np.zeros(1), phase='train',
                 num1=0, num2=0,
                 data="data/",
                 numTx=80,
                 thresh=0.,
                 transform=transforms.ToTensor()):

        if maps.size == 1:
            self.maps = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(2024)
            np.random.shuffle(self.maps)
        else:
            self.maps = maps

        self.data = data
        self.numTx = numTx
        self.thresh = thresh
        self.transform = transform
        self.height = 256
        self.width = 256
        self.num1 = num1
        self.num2 = num2

        if phase == 'train':
            self.num1 = 0
            self.num2 = 500
        elif phase == 'val':
            self.num1 = 501
            self.num2 = 600
        elif phase == 'test':
            self.num1 = 601
            self.num2 = 700

        self.simulation = self.data + "gain/"
        self.build = self.data + "png/buildingsWHeight/"
        self.antenna = self.data + "png/antennasWHeight/"
        self.free_pro = self.data + "png/free_propagation/"

    def __len__(self):
        return (self.num2 - self.num1) * self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map = self.maps[idxr + self.num1]

        name1 = str(dataset_map) + ".png"

        name2 = str(dataset_map) + "_" + str(idxc) + ".png"

        # Loading Build
        builds = os.path.join(self.build, name1)

        # To Numpy
        arr_build = np.asarray(io.imread(builds))

        # Loading Antenna
        antennas = os.path.join(self.antenna, name2)

        # To Numpy
        arr_antenna = np.asarray(io.imread(antennas))

        # Loading Path Loss Maps
        free_pro = os.path.join(self.free_pro, name2)
        arr_free_pro = np.asarray(io.imread(free_pro))

        # Loading Target
        target = os.path.join(self.simulation, name2)

        # To Numpy
        arr_target = np.asarray(io.imread(target))

        # Threshold Transfer
        if self.thresh >= 0:
            arr_target = arr_target / 255
            mask = arr_target < self.thresh
            arr_target[mask] = self.thresh
            arr_target = arr_target - self.thresh * np.ones(np.shape(arr_target))
            arr_target = arr_target / (1 - self.thresh)

        # 转张量
        arr_builds = self.transform(arr_build).type(torch.float32)
        arr_antennas = self.transform(arr_antenna).type(torch.float32)
        arr_free_pros = self.transform(arr_free_pro).type(torch.float32)
        arr_targets = self.transform(arr_target).type(torch.float32)

        return arr_builds, arr_antennas, arr_targets, name2

class loader_3D1(Dataset):

    def __init__(self, phase='train',
                 data="data/",
                 numTx=80,
                 thresh=0.,
                 rank=3,
                 transform=transforms.ToTensor()):

        self.data = data
        self.phase = phase
        self.numTx = numTx
        self.thresh = thresh
        self.transform = transform
        self.height = 256
        self.width = 256

        if phase == 'train':
            if rank == 1:
                self.num = self.data + "rank1.npy"
                self.len = 277

            elif rank == 2:
                self.num = self.data + "rank2.npy"
                self.len = 233

            elif rank == 3:
                self.num = self.data + "rank3.npy"
                self.len = 163

        elif phase == 'val':
            self.num = self.data + "rank4.npy"
            self.len = 27

        elif phase == 'test':
            self.num = self.data + "rank4.npy"
            self.len = 27

        self.simulation = self.data + "gain/"
        self.build = self.data + "png/buildingsWHeight/"
        self.free_pro = self.data + "png/free_propagation/"
        self.antenna = self.data + "png/antennasWHeight/"

    def __len__(self):

        return self.len * self.numTx

    def __getitem__(self, idx):

        self.maps = np.load(self.num).tolist()

        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map = self.maps[idxr]

        name1 = str(dataset_map) + ".png"

        name2 = str(dataset_map) + "_" + str(idxc) + ".png"

        builds = os.path.join(self.build, name1)

        arr_build = np.asarray(io.imread(builds))

        free_pro = os.path.join(self.free_pro, name2)

        arr_free_pro = np.asarray(io.imread(free_pro))

        antennas = os.path.join(self.antenna, name2)

        arr_antenna = np.asarray(io.imread(antennas))

        target = os.path.join(self.simulation, name2)

        arr_target = np.asarray(io.imread(target))

        # 阈值变换
        if self.thresh >= 0:
            arr_target = arr_target / 255
            mask = arr_target < self.thresh
            arr_target[mask] = self.thresh
            arr_target = arr_target - self.thresh * np.ones(np.shape(arr_target))
            arr_target = arr_target / (1 - self.thresh)

        # 转张量
        arr_builds = self.transform(arr_build.copy()).type(torch.float32)
        arr_antennas = self.transform(arr_antenna.copy()).type(torch.float32)
        arr_free = self.transform(arr_free_pro.copy()).type(torch.float32)
        arr_targets = self.transform(arr_target.copy()).type(torch.float32)

        return arr_builds, arr_antennas, arr_targets, name2


def test():
    dataset = loader_3D(phase='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for w, x, y, z, o in loader:
        print(w.shape, x.shape, y.shape, z.shape)

if __name__ == "__main__":
    test()






