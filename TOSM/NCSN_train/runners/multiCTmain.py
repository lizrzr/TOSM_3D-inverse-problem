import sys
import os
import numpy as np
from numpy import matlib
from .loadData import loadData
from scipy.interpolate import interpn
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
import cv2
import time
import astra
import scipy.io as sio
from .datamaking import datamaking_test
from torch.utils.data import DataLoader
from .compose import compose
import shutil
from PIL import Image

class xiaoshuRecon():
    def __init__(self):
        self.projGeom = astra.create_proj_geom('cone', 0.127, 0.127, 1024, 1024, np.linspace(0, 2 * np.pi, 720), 145.1,
                                               355.1)
        self.volGeom = astra.create_vol_geom(512, 512, 512, (-512 / 2) * 37.69 / 512, (512 / 2) * 37.69 / 512,
                                             (-512 / 2) * 37.69 / 512,
                                             (512 / 2) * 37.69 / 512, (-512 / 2) * 37.69 / 512, (512 / 2) * 37.69 / 512)
        self.projGeom1 = astra.create_proj_geom('cone', 0.127, 0.127, 1024, 1024, np.linspace(0, 2 * np.pi, 29), 145.1,
                                               355.1)

    def circle_shift(self, img, x, y):
        img = np.roll(img, x, axis=0)
        img = np.roll(img, y, axis=1)
        return img

    def loadPro(self):
        path = './data/'
        prjS = np.empty(shape=(1024, 0, 1024))
        for i in range(720):
            name = path + str(i).zfill(5) + '.tif'
            prj = np.array(Image.open(name))
            prj = self.circle_shift(prj, 0, 4)

            prj = np.log(prj)
            left_columes = prj[:, :10]
            right_columes = prj[:, :-10]
            left_columes_mean = np.mean(left_columes, axis=1)
            right_columes_mean = np.mean(right_columes, axis=1)
            prj = (left_columes_mean+right_columes_mean)/2 - prj
            prj[prj < 0] = 0
            prj = prj[:, np.newaxis, :]
            prjS = np.append(prjS, prj, axis=1)
        return prjS

    def FDK_Cone(self, PRO):
        proj_geom = self.projGeom
        vol_geom = self.volGeom
        rec_id = astra.data3d.create('-vol', vol_geom)
        proj_id = astra.data3d.create('-sino', proj_geom, PRO)
        cfg = astra.astra_dict('FDK_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data3d.get(rec_id)
        pro = astra.data3d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)

        return rec

    def SIRT_CONE(self, PRO, VOL):
        proj_geom = self.projGeom1
        vol_geom = self.volGeom
        if VOL is None:
            rec_id = astra.data3d.create('-vol', vol_geom)
        else:
            rec_id = astra.data3d.create('-vol', vol_geom, VOL)
        proj_id = astra.data3d.create('-sino', proj_geom, PRO)
        cfg = astra.astra_dict('SIRT3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id,20)
        rec = astra.data3d.get(rec_id).T
        pro = astra.data3d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)

        return rec

    def FP_CONE(self, VOL):
        proj_geom = self.projGeom1
        vol_geom = self.volGeom
        rec_id = astra.data3d.create('-vol', vol_geom, VOL)
        proj_id = astra.data3d.create('-sino', proj_geom)
        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['VolumeDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id
        cfg['option'] = {'VoxelSuperSampling': 2}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data3d.get(rec_id).T
        pro = astra.data3d.get(proj_id)
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)

        return pro