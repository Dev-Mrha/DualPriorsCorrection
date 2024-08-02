"""
This file runs the main training/val loop
"""
import os
import json
import math
import sys
import pprint
import torch
from argparse import Namespace

sys.path.append(".")
sys.path.append("..")

from options.train_options import TrainOptions
from model.facenet import Net


def main():
	opts = TrainOptions().parse()
	previous_train_ckpt = None
	if opts.resume_training_from_ckpt:
		opts, previous_train_ckpt = load_train_checkpoint(opts)
	else:
		create_initial_experiment_dir(opts)

	coach = Net(opts, previous_train_ckpt)
	coach.train()


def load_train_checkpoint(opts):
	train_ckpt_path = opts.resume_training_from_ckpt
	previous_train_ckpt = torch.load(opts.resume_training_from_ckpt, map_location='cpu')
	new_opts_dict = vars(opts)
	opts = previous_train_ckpt['opts']
	opts['resume_training_from_ckpt'] = train_ckpt_path
	update_new_configs(opts, new_opts_dict)
	pprint.pprint(opts)
	opts = Namespace(**opts)
	if opts.sub_exp_dir is not None:
		sub_exp_dir = opts.sub_exp_dir
		opts.exp_dir = os.path.join(opts.exp_dir, sub_exp_dir)
		create_initial_experiment_dir(opts)
	return opts, previous_train_ckpt


def create_initial_experiment_dir(opts):
	if os.path.exists(opts.exp_dir):
		raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)


def update_new_configs(ckpt_opts, new_opts):
	for k, v in new_opts.items():
		if k not in ckpt_opts:
			ckpt_opts[k] = v
	if new_opts['update_param_list']:
		for param in new_opts['update_param_list']:
			ckpt_opts[param] = new_opts[param]


if __name__ == '__main__':
	main()

"""
python train.py \
--dataset_type facecnet_data \
--exp_dir experiment/ \
--start_from_latent_avg \
--val_interval 40000 \
--save_interval 20000 \
--max_steps 200000 \
--image_interval 1000 \
--stylegan_size 512 \
--stylegan_weights pretrained_models/stylegan2.pt \
--workers 4 \
--batch_size 4 \
--test_batch_size 8 \
--test_workers 4 \
--checkpoint_path pretrained_models/0412_best_model.pt \
--load_model pretrained_model/facenet_lq_best_model.pt 
"""

"""
python train.py --dataset_type facecnet_data --exp_dir experiment/ --start_from_latent_avg --val_interval 40000 --max_steps 200000 --stylegan_size 512 --stylegan_weights pretrained_models/stylegan2-ffhq-config-f.pt --workers 2 --batch_size 2 --test_batch_size 2 --test_workers 2 --checkpoint_path pretrained_models/best_model.pt
"""
