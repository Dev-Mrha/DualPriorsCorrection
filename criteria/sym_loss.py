import torch.nn as nn
import torch
import numpy as np


class SymLoss(nn.Module):
    def __init__(self):
        super(SymLoss, self).__init__()

    def forward(self, flow):
        flow = np.array(flow.cpu())
        flow_rl = flow[:, :, :, ::-1].copy()
        flow_ud = flow[:, :, ::-1, :].copy()
        flow_rl = torch.tensor(flow_rl)
        flow_ud = torch.tensor(flow_ud)
        loss_rl = torch.norm(flow_rl - flow, p=2, dim=1).mean()
        loss_ud = torch.norm(flow_ud - flow, p=2, dim=1).mean()
        return loss_rl + loss_ud
