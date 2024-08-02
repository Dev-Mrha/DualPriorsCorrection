import os
import random

import cv2
import numpy as np
import torch
import Imath, OpenEXR
from PIL import Image
import torchvision.transforms as transforms
from .randaugment import RandAugmentMC
import torch.nn.functional as F
import face_alignment

class TransformFixMatch(object):
    def __init__(self):
        self.common = transforms.Compose([transforms.RandomHorizontalFlip()])
        self.strong = transforms.Compose([RandAugmentMC(n=2, m=9)])

    def __call__(self, x):
        weak = self.common(x)
        strong = self.strong(weak)
        return weak, strong


class LabeledData():
    def __init__(self, data_dir, input_w, input_h, down_sample_factor=8):
        self.input_w = input_w
        self.input_h = input_h
        self.datas_infos = []
        self.down_sample_factor = down_sample_factor
        self.transform = transforms.Compose([
            transforms.Resize((input_h, input_w)),
            transforms.ToTensor()])
        # print(self.transform)
        self.datas_infos = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) \
                            if os.path.isfile(os.path.join(data_dir, filename)) and filename.endswith("p.jpg")]
        self.datas_num = len(self.datas_infos)
        self.detecter = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=self.device)
        print("number of datas:", self.datas_num)
        print("init data finished")

    def __len__(self):
        return self.datas_num

    def _exr_flow(self, pth):
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        img_exr = OpenEXR.InputFile(pth)
        dw = img_exr.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        imgx = Image.frombytes("F", size, img_exr.channel('Y', pt))
        imgx = np.asarray(imgx)
        imgx = imgx.reshape(size[1], size[0])
        imgx = torch.tensor(imgx)
        imgx = imgx.unsqueeze(0).unsqueeze(0)
        return imgx

    def __getitem__(self, index):
        img_path = self.datas_infos[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.transform(img)

        img_map_x_path = img_path.replace('.jpg', '_orig2line_mapx.exr')
        img_map_y_path = img_path.replace('.jpg', '_orig2line_mapy.exr')

        flow_map_x = self._exr_flow(img_map_x_path)  # read flow map x direction
        flow_map_y = self._exr_flow(img_map_y_path)  # read flow map y direction
        flow_map_h, flow_map_w = flow_map_x.shape[-2:]

        scale_x = self.input_w // self.down_sample_factor / flow_map_w
        scale_y = self.input_h // self.down_sample_factor / flow_map_h

        flow_map_x = F.interpolate(flow_map_x,
                                (self.input_h // self.down_sample_factor, self.input_w // self.down_sample_factor)).squeeze(0)
        flow_map_y = F.interpolate(flow_map_y,
                                (self.input_h // self.down_sample_factor, self.input_w // self.down_sample_factor)).squeeze(0)
        flow_map_x *= scale_x
        flow_map_y *= scale_y

        gt_path = img_path.replace('.jpg', '_line.jpg')
        gt_img = Image.open(gt_path)
        gt_img = gt_img.convert('RGB')
        gt_img = self.transform(gt_img)
        # facemask = (facemask / 255.0)

        return img, flow_map_x, flow_map_y, gt_img
