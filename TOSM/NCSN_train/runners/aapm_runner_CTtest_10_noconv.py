import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from scipy.sparse.linalg import spsolve
import sys
import math
from ..models.cond_refinenet_dilated_noconv import CondRefineNetDilated
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
from scipy.ndimage.filters import laplace
import matplotlib.pyplot as plt
# from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from .multiCTmain import reconstruct,xiaoshuRecon
import glob
import h5py
import time
from skimage import img_as_float, img_as_ubyte, io
# from scipy.misc import imread,imsave
from scipy.linalg import norm, orth
from scipy.stats import poisson
import pydicom
from skimage.transform import radon, iradon
import odl
import cvxpy as cp
import scipy.fftpack as fft

plt.ion()
savepath = './result/'
__all__ = ['Aapm_Runner_CTtest_10_noconv']


import numpy as np
from scipy.optimize import minimize

class Aapm_Runner_CTtest_10_noconv():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def write_images(self, x, image_save_path):
        x = np.array(x, dtype=np.uint8)
        cv2.imwrite(image_save_path, x)

    def test(self):
        # 2023.03.06 lizrzr
        SIRTiter = xiaoshuRecon()
        PRJS = SIRTiter.loadPro()
        img3d = SIRTiter.FDK_Cone(PRJS)
        savemat('img3d.mat', {'img3d': img3d})
        PRJSSPARSE = SIRTiter.FP_CONE(img3d)
        plt.imshow(img3d[:, :, 256], cmap='gray')
        plt.show()
        states = torch.load(os.path.join(self.args.log, 'checkpoint_62500.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet, device_ids=[0])
        scorenet.load_state_dict(states[0])
        scorenet.eval()

        x = img3d

        maxdegrade = np.zeros(512)
        for i in range(512):
            maxdegrade[i] = np.max(img3d[:, :, i])

        x0 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 512, 512, 512])).uniform_(-1, 1))
        x01 = x0.cuda()
        step_lr = 0.6 * 0.00003
        sigmas = np.exp(np.linspace(np.log(0.8), np.log(0.01), 12))
        n_steps_each = 10
        max_psnr = 0
        max_ssim = 0
        min_hfen = 100
        start_start = time.time()
        for idx, sigma in enumerate(sigmas):
            start_out = time.time()
            print(idx)
            lambda_recon = 1. / sigma ** 2
            labels = torch.ones(1, device=x0.device) * idx
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            print('sigma = {}'.format(sigma))
            for step in range(n_steps_each):
                start_in = time.time()
                if step == -1:
                    noise1 = torch.rand_like(x0).cpu().detach() * np.sqrt(step_size * 2)
                    with torch.no_grad():
                        for i in range(512):
                            grad1[:, :, i, :, :] = scorenet(x01[:, :, i, :, :], labels).detach()
                    x0 = x0 + step_size * grad1
                    x01 = x0 + noise1
                    x0 = np.array(x0.cpu().detach(), dtype=np.float32)
                    x0 = np.expand_dims(x0, 2)
                    x0 = np.tile(x0, (1, 1, 512, 1, 1))

                    x1 = np.squeeze(x0)
                    x1 = np.mean(x1, axis=0)

                    x01 = torch.unsqueeze(x01, 2)
                    x01 = x01.repeat(1, 1, 512, 1, 1)
                    print(x01.shape)
                else:
                    noise1 = torch.rand_like(x0).cpu().detach() * np.sqrt(step_size * 2)
                    grad1 = np.zeros([1, 10, 512, 512, 512])
                    grad1 = torch.from_numpy(grad1)
                    grad2 = np.zeros([1, 10, 512, 512, 512])
                    grad2 = torch.from_numpy(grad2)
                    grad3 = np.zeros([1, 10, 512, 512, 512])
                    grad3 = torch.from_numpy(grad3)
                    with torch.no_grad():
                        for i in range(512):
                            grad1[:, :, i, :, :] = scorenet(x01[:, :, i, :, :], labels).detach()
                            grad2[:, :, :, :, i] = scorenet(x01[:, :, :, :, i], labels).detach()
                            grad3[:, :, :, i, :] = scorenet(x01[:, :, :, i, :], labels).detach()

                    x0 = x0 + (step_size * grad1 + step_size * grad2 + step_size * grad3) / 3
                    x01 = x0 + noise1
                    x01 = torch.tensor(x01.cuda(), dtype=torch.float32)
                    x0 = np.array(x0.cpu().detach(), dtype=np.float32)
                    x1 = np.squeeze(x0)

                    x1 = np.mean(x1, axis=0)


                x1max = np.zeros([512, 512, 512])
                for i in range(512):
                    x1max[i, :, :] = x1[i, :, :] * maxdegrade[i]

                sum_diff = x - x1max
                x_new = 0.8 * SIRTiter.SIRT_CONE(VOL=x.copy(),PRO=PRJSSPARSE) + 0.2 * (
                        x - sum_diff)
                if step % 1 == 0:
                    plt.title(step, fontsize=30)
                    plt.imshow(x_new[:, :, 256], cmap='gray')
                    plt.show()
                    plt.title(step, fontsize=30)
                    plt.imshow(x_new[:, 256, 100:400], cmap='gray')
                    plt.show()
                    plt.imshow(x_new[256, :, :], cmap='gray')
                    plt.show()

                x = x_new

                x_rec = np.transpose(x.copy(), (2, 0, 1))
                for i in range(512):
                    x_rec[i, :, :] = x_rec[i, :, :] / maxdegrade[i]
                # x_rec = x_rec / maxdegrade
                if step == n_steps_each - 1:
                    savemat('./' + str(idx) + 'sirt_0.5_x_rec.mat', {'x_rec': x_rec})
                end_in = time.time()
                print("inner loop:%.2fs" % (end_in - start_in))

                print("current {} step".format(step))
                x_mid = np.zeros([1, 10, 512, 512, 512], dtype=np.float32)
                x_rec = np.clip(x_rec, 0, 1)
                x_rec = np.expand_dims(x_rec, 0)
                x_mid_1 = np.tile(x_rec, [10, 1, 1, 1])
                x_mid[0, :, :, :, :] = x_mid_1
                x0 = torch.tensor(x_mid, dtype=torch.float32)
            end_out = time.time()
            print("outer iter:%.2fs" % (end_out - start_out))

        plt.ioff()
        end_end = time.time()
        savemat('x_rec.mat', {'x_rec': x_rec})
        print("PSNR:%.2f" % (max_psnr), "SSIM:%.2f" % (max_ssim))
        print("total time:%.2fs" % (end_end - start_start))
