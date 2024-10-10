from model.unet import U_Net_Line, U_Net_Shape
from model.psp import pSp
import torch.nn as nn
from PIL import Image
import torch
import torchvision.transforms as tf
import torch.nn.functional as F
import face_alignment
import numpy as np
import scipy
import torchvision.transforms as transforms
from utils import utils_faces
from torchvision.utils import save_image
import cv2
import dlib
import sys
import os
from .retinaface.retinaface_detection import RetinaFaceDetection
from .bisenet.model import BiSeNet
from .parsenet import  ParseNet
import time


class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.device = device
        self.linenet = U_Net_Line(in_ch=3, out_ch=2).to(self.device)
        self.shapenet = U_Net_Shape(in_ch=3, out_ch=2).to(self.device)
        self.psp = pSp(self.device).to(self.device)
        self.black = np.zeros((256, 256), np.float32)
        self.transform = transforms.Compose([transforms.Resize((384, 512)), transforms.ToTensor()])
        self.tf_face = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        self.facedetector = RetinaFaceDetection('./')
        self.reference_5pts = utils_faces.get_reference_facial_points((256, 256), 0.25, (0, 0), True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.parsenet = ParseNet(512, 512, 32, 64, 19, norm_type='bn', relu_type='LeakyReLU', ch_range=[32, 256])
        xx = torch.arange(0, 256).view(1,-1).repeat(256,1)
        yy = torch.arange(0, 256).view(-1,1).repeat(1,256)
        xx = xx.view(1,1,256,256).repeat(1,1,1,1)
        yy = yy.view(1, 1, 256, 256).repeat(1, 1, 1, 1)
        self.grid = torch.cat((xx,yy), dim=1).float()
        self.grid = self.grid.permute(0,2,3,1).to(self.device)

    def load_ckpt(self, pSp_ckpt, line_ckpt, shape_ckpt):
        line = torch.load(line_ckpt, map_location=self.device)
        self.linenet.load_state_dict(line['state_dict'])
        self.psp.load_weights(pSp_ckpt)
        shape = torch.load(shape_ckpt, map_location=self.device)
        self.shapenet.load_state_dict(shape['state_dict'])

    def color_parse_map(self, tensor):
        """
        input: tensor or batch tensor
        return: colorized parsing maps
        """
        MASK_COLORMAP = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204],
                         [0, 255, 255],
                         [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153],
                         [0, 0, 204],
                         [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]  # 19

        label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
                      'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']  # 18

        remove_list = [18, 17, 16, 14]  # cloth, neck, neck_l, hat
        # remove_list = [18]  # cloth

        if len(tensor.shape) < 4:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] > 1:
            tensor = tensor.argmax(dim=1)

        tensor = tensor.squeeze(1).data.cpu().numpy()
        color_maps = []
        for t in tensor:
            tmp_img = np.zeros(tensor.shape[1:] + (3,))
            for idx, color in enumerate(MASK_COLORMAP):
                if idx not in remove_list:
                    tmp_img[t == idx] = color
            color_maps.append(tmp_img.astype(np.uint8))
        return color_maps

    def remove_bg(self, face):
        parse_net = self.parsenet
        parse_net.eval()
        parse_net.load_state_dict(torch.load('./pretrained_models/parse_multi_iter_90000.pth'))
        face = face.copy()
        face = cv2.resize(face, (512,512))
        img_tensor = self.trans(face).unsqueeze(0).to(self.device)
        with torch.no_grad():
            parse_map, _ = parse_net(img_tensor)

        parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()

        color_maps = self.color_parse_map(parse_map_sm)

        img_np = color_maps[0]
        img_np = np.mean(img_np, axis=2)
        img_np[img_np > 0] = 1
        img_np = cv2.resize(img_np, (256,256))
        return img_np
    
    def combine_flow(self, flow, face_flow, bbox, move_x, move_y):
        face_flow = (face_flow + 1.0) * 255 / 2.0
        face_flow = face_flow - self.grid
        height, width = int(bbox[2]) - int(bbox[0]), int(bbox[3]) - int(bbox[1])
        face_flow[:, :, :, 0] = face_flow[:, :, :, 0].clone() / 256 * width
        face_flow[:, :, :, 1] = face_flow[:, :, :, 1].clone() / 256 * height
        face_flow = F.interpolate(face_flow.permute(0, 3, 1, 2), (width, height)).permute(0, 2, 3, 1)
        face_flow = face_flow.squeeze()
        g_face_flow = np.zeros_like(flow)
        for i in range(width):
            for j in range(height):
                u, v = int(i + face_flow[i][j][0] + bbox[1]+move_x), int(j + face_flow[i][j][1] + bbox[0]+move_y)
                if 0 < u and u < 384 and 0 < v and v < 512:
                    g_face_flow[0][u][v][0] = face_flow[i][j][0]
                    g_face_flow[0][u][v][1] = face_flow[i][j][1]
        i_indices, j_indices = np.meshgrid(np.arange(384), np.arange(512), indexing='ij')
        u_indices = (i_indices + flow[0, :, :, 0]).astype(int)
        v_indices = (j_indices + flow[0, :, :, 1]).astype(int)
        mask = (u_indices > 0) & (u_indices < 384) & (v_indices > 0) & (v_indices < 512)
        flow[0][mask] += g_face_flow[0][u_indices[mask], v_indices[mask]]
        return flow


    def forward(self, img, return_flow=False):
        ori_w, ori_h = img.size
        input = img.convert('RGB')
        img = np.array(img)
        input = self.transform(input).unsqueeze(0)
        input = input.to(self.device)
        f_mid = self.linenet(input)
        f_mid = f_mid.detach().cpu().squeeze().numpy()
        f_mid = f_mid.transpose(1, 2, 0)
        predflow_x, predflow_y = f_mid[:, :, 0], f_mid[:, :, 1]

        scale_x = ori_w / predflow_x.shape[1]
        scale_y = ori_h / predflow_x.shape[0]
        predflow_x = cv2.resize(predflow_x, (ori_w, ori_h)) * scale_x
        predflow_y = cv2.resize(predflow_y, (ori_w, ori_h)) * scale_y
        ys, xs = np.mgrid[:ori_h, :ori_w]
        mesh_x = predflow_x.astype("float32") + xs.astype("float32")
        mesh_y = predflow_y.astype("float32") + ys.astype("float32")
        # img_mid = cv2.remap(img, mesh_x, mesh_y, cv2.INTER_LINEAR)

        facebs, landms = self.facedetector.detect(img)
        full_mask = np.zeros((ori_h, ori_w), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        ori_img = np.copy(img)
        out_msk = np.zeros_like(ori_img)

        final_flow = f_mid.copy()
        final_flow = np.stack([final_flow])

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            facial5points = np.reshape(facial5points, (2, 5))
            face, tfm_inv = utils_faces.warp_and_crop_face(img, facial5points, crop_size=(256, 256),
                                                           reference_pts=self.reference_5pts)
            tmp_mask = self.remove_bg(face)
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (ori_w, ori_h), flags=3)
            black = self.black.astype(np.float32)
            black = cv2.warpAffine(black, tfm_inv, (ori_w, ori_h), flags=3)
            black = black[:, :, np.newaxis]
            ori_img[np.where(tmp_mask > 0)] = black[np.where(tmp_mask > 0)] * 255
            out_msk[np.where(tmp_mask > 0)] = 255
        out_msk = cv2.remap(out_msk, mesh_x, mesh_y, cv2.INTER_LINEAR)
        nw_img_mid = cv2.remap(ori_img, mesh_x, mesh_y, cv2.INTER_LINEAR)

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            
            facial5points = np.reshape(facial5points, (2, 5))
            mat = np.float32([[1, 0, 0], [0, 1, 0]])
            nose_x, nose_y = facial5points[0][2], facial5points[1][2]
            nose_x_pred, nose_y_pred = nose_x - predflow_x[int(nose_y), int(nose_x)], nose_y - predflow_y[int(nose_y), int(nose_x)]
            mat[0][2] = nose_x_pred - nose_x
            mat[1][2] = nose_y_pred - nose_y

            face, tfm_inv = utils_faces.warp_and_crop_face(img, facial5points, crop_size=(256, 256),
                                                           reference_pts=self.reference_5pts)
            face = self.tf_face(Image.fromarray(face)).to(self.device).unsqueeze(0)

            e4eimg, features = self.psp(face)
            flow_face = self.shapenet(face, features)

            flow_face = flow_face.permute(0, 2, 3, 1)
            if return_flow:
                final_flow = self.combine_flow(final_flow, flow_face, faceb, mat[0][2], mat[1][2])
            out_face = F.grid_sample(face, flow_face, mode='bilinear', align_corners=True)
            out_face = out_face.squeeze().permute(1, 2, 0).flip(2)
            out_face = out_face.cpu().numpy()

            parse_mask = self.remove_bg(out_face)

            parse_mask = cv2.warpAffine(parse_mask, tfm_inv, (ori_w, ori_h), flags=3)
            parse_mask = cv2.warpAffine(parse_mask, mat, (ori_w, ori_h), flags=3)
            parse_mask = cv2.dilate(parse_mask, self.kernel, iterations=2)

            tmp_img = cv2.warpAffine(out_face, tfm_inv, (ori_w, ori_h), flags=3)
            tmp_img = cv2.warpAffine(tmp_img, mat, (ori_w, ori_h), flags=3)
            tmp_img = tmp_img[:, :, ::-1]

            mask = parse_mask - full_mask
            full_mask[np.where(parse_mask > 0)] = parse_mask[np.where(parse_mask > 0)]
            full_img[np.where(parse_mask > 0)] = tmp_img[np.where(parse_mask > 0)] * 255
            out_msk[np.where(parse_mask > 0)] = 0

        full_mask = full_mask[:, :, np.newaxis]
        ret_img = cv2.convertScaleAbs(nw_img_mid * (1 - full_mask) + full_img * full_mask)
        out_msk = cv2.dilate(out_msk, self.kernel, iterations=3)
        out_msk[np.where(out_msk > 0)] = 1
        if return_flow:
            return ret_img, f_mid, final_flow[0]
        else:
            return ret_img, out_msk 
