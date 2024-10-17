import numpy as np
import torch
import cv2
import Imath
import OpenEXR
from PIL import Image
import os
from tqdm import tqdm
from torch.nn import functional as F
import argparse

def distort(img_path, f_flow):
    img = cv2.imread(img_path)
    f_flow = f_flow.numpy().astype('float32')
    predflow_x, predflow_y = f_flow[:, :, 0], f_flow[:, :, 1]
    scale_x = 1024 / predflow_x.shape[1]
    scale_y = 1024 / predflow_x.shape[0]
    predflow_x = cv2.resize(predflow_x, (1024, 1024)) * scale_x
    predflow_y = cv2.resize(predflow_y, (1024, 1024)) * scale_y
    ys, xs = np.mgrid[:1024, :1024]
    mesh_x = predflow_x.astype("float32") + xs.astype("float32")
    mesh_y = predflow_y.astype("float32") + ys.astype("float32")
    pred_out = cv2.remap(img, mesh_x, mesh_y, cv2.INTER_LINEAR)
    save_path = img_path.replace("face_dataset", "face_dataset_distorted")
    cv2.imwrite(save_path, pred_out)

def loop_gen(img_list):
    mx = 875
    cnt = 0
    flow_dir = "flow/"
    for img_path in img_list:
        img = cv2.imread(img_path)
        f_flow = torch.load(os.path.join(flow_dir ,str(cnt) + ".pth"))
        distort(img_path, f_flow)
        cnt += 1
        if cnt >= mx:
            cnt = 0

def exr_flow(pth):
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    img_exr = OpenEXR.InputFile(pth)
    dw = img_exr.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    imgx = Image.frombytes("F", size, img_exr.channel('Y', pt))
    imgx = np.asarray(imgx)
    imgx = imgx.reshape(size[1], size[0])
    imgx = torch.tensor(imgx)
    imgx = imgx.unsqueeze(2)
    return imgx

def read_exr(pth):
    img_map_x_path = pth+"_orig2line_mapx.exr"
    img_map_y_path = pth+"_orig2line_mapy.exr"
    flow_map_x = exr_flow(img_map_x_path)  # read flow map x direction
    flow_map_y = exr_flow(img_map_y_path) # read flow map y direction
    flow = torch.cat((flow_map_x, flow_map_y), dim=2)
    return flow


def gen_flow(flow_dir):
    w = 1920
    h = 2560
    s = 64 # in order to save the memory, we only save the flow of the patches with stride 64
    
    flow = read_exr("../train_4_3/train_4_3/train_4_3/0003_a9s_1p")# the real undistortion flow from the dataset, you can replace it with your own flow
    flow = flow.permute(2, 0, 1).unsqueeze(0)
    flow = F.interpolate(flow, (w, h))
    flow = flow.squeeze().permute(1,2,0)
    flow = flow.to(torch.float32)

    # randomly choose the centers of the patches
    centers = []
    for i in range(64, w - 256 + 1 - 64, s):
        for j in range(64, h - 256 + 1 - 64, s):
            centers.append([i, j])

    for k in tqdm(range(len(centers))):
        cx, cy = centers[k]
        ff = flow[cx: cx + 256, cy: cy + 256, :]
        g = torch.zeros_like(ff)
        for i in range(256):
            for j in range(256):
                u, v = i + int(ff[i, j, 0]), j + int(ff[i, j, 1])
                if 0 <= u < 256 and 0 <= v < 256:
                    g[u, v, 0] = -ff[i, j, 0]
                    g[u, v, 1] = -ff[i, j, 1]
        torch.save(g, os.path.join(flow_dir ,str(k) + ".pth"))

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--option', type=['image', 'flow'], default='image')
    argparse.add_argument('--dir', type=str, default='../face_dataset/')
    args = argparse.parse_args()
    
    if args.option == 'image':
        img_list = os.listdir(args.dir)
        img_list = [dir + img for img in img_list]
        loop_gen(img_list)
    else:
        gen_flow(args.dir)
    
    