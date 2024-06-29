import torch
import os
import numpy as np
from PIL import Image
from models import modules
from dataloaders import loaders
from torch.utils.data import Dataset, DataLoader

def main_worker():
    # loading model and parameter
    device = torch.device("cuda:0")
    model = modules.REM_Net()
    model.load_state_dict(torch.load('model_result/Best_weight.pt'))
    model.to(device)

    # loading dataloader
    test_data = loaders.loader_3D(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=2)

    interation = 0
    err1 = []
    err2 = []

    # testing
    # # if not using path loss map
    for build, antenna, target, img_name in test_dataloader:
    # # if using path loss map
    # for build, antenna, sample, target, img_name in test_dataloader:

        interation += 1
        # # if not using path loss map
        build, antenna, target = build.cuda(), antenna.cuda(), target.cuda()
        # # if using path loss map
        #build, antenna, sample, target = sample.cuda(), mask.cuda(), sample.cuda(), target.cuda()

        with torch.no_grad():

            # # if not using path loss map
            pre = model(build, antenna)

            # # if using path loss map
            # pre = model(build, antenna, antenna)

        # predict
        test = torch.tensor([item.cpu().detach().numpy() for item in pre]).cuda()
        test = test.squeeze(0)
        test = test.squeeze(0)
        im1 = test.cpu().numpy()
        im2 = test.cpu().numpy()*255
        predict1 = Image.fromarray(im2.astype(np.uint8))

        # target
        test1 = torch.tensor([item.cpu().detach().numpy() for item in target]).cuda()
        test1 = test1.squeeze(0)
        test1 = test1.squeeze(0)
        im = test1.cpu().numpy()
        image = test1.cpu().numpy()*255
        images = Image.fromarray(image.astype(np.uint8))


        # rmse
        rmse1 = np.sqrt(np.mean((im - im1) ** 2))
        err1.append(rmse1)
        # nmse
        nmse1 = np.mean((im - im1) ** 2)/np.mean((0 - im) ** 2)
        err2.append(nmse1)

        # saving
        image_name = os.path.basename(img_name[0]).split('.')[0]
        images.save(os.path.join("image_result", f'{image_name}_target.png'))
        predict1.save(os.path.join("image_result", f'{image_name}_predict1.png'))
        print(f'saving to {os.path.join("image_result", image_name)}', "RMSE:", rmse1, "NMSE:", nmse1)

        # total = 8000
        if interation >= 8000:
            break
    rmse_err = sum(err1)/len(err1)
    nmse_err = sum(err2) / len(err2)


    print('一阶段测试集均方根误差：', rmse_err)
    print('一阶段测试集归一化均方误差：', nmse_err)

if __name__ == '__main__':
 main_worker()