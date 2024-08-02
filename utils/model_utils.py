import torch
import argparse
from model.model import Net

def setup_model(checkpoint_path, device='cuda:0'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = argparse.Namespace(**opts)

    net = Net(opts)
    net.net.load_state_dict(ckpt['state_dict'])
    net.net.eval()
    net.net = net.net.to(device)
    return net, opts
