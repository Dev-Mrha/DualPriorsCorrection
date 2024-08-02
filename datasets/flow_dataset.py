from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import torch
from torchvision.transforms import Resize


class FlowDataset(Dataset):

	def __init__(self, source_root, target_root, flow_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.flow_paths = sorted(data_utils.make_dataset(flow_root, txt=True))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def readFlow(self, fn):
		with open(fn) as f:
			w = np.fromfile(f, np.int32, count=1)
			h = np.fromfile(f, np.int32, count=1)
			data = np.fromfile(f, np.float32, count = 2 * int(w) * int(h))
			return np.resize(data, (int(w), int(h), 2))

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')

		flow_path = self.flow_paths[index]
		flow = np.loadtxt(flow_path)

		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)

		flow1 = torch.tensor(flow[:256][:]).reshape(256, 256, 1)
		flow2 = torch.tensor(flow[256:][:]).reshape(256, 256, 1)
		flow = torch.cat((flow1, flow2), dim=2)

		return from_im, to_im, flow
