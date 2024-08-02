from abc import abstractmethod
import torchvision.transforms as transforms


class TransformsConfig(object):

	def __init__(self, opts):
		self.opts = opts

	@abstractmethod
	def get_transforms(self):
		pass


class EncodeTransforms(TransformsConfig):

	def __init__(self, opts):
		super(EncodeTransforms, self).__init__(opts)

	def get_transforms(self):
		transforms_dict = {
			'transform_gt_train': transforms.Compose([
				transforms.Resize((384, 256)),
				# transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor()]),
				# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_source': transforms.Compose([
				# transforms.Resize((384, 256)),
				# transforms.CenterCrop((1200, 1200)),
				transforms.Resize((512, 384)),
				transforms.ToTensor()]),
				# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_test': transforms.Compose([
				transforms.Resize((384, 256)),
				transforms.ToTensor()]),
				# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
			'transform_inference': transforms.Compose([
				transforms.Resize((384, 256)),
				transforms.ToTensor()])
		}
		return transforms_dict

