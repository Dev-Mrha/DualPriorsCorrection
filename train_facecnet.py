import os
import json
import sys
import pprint
from argparse import Namespace

sys.path.append(".")

from options.train_options import TrainOptions
from model.facenet import Net


def create_initial_experiment_dir(opts):
	if os.path.exists(opts.exp_dir):
		raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)


if __name__ == '__main__':
	opts = TrainOptions().parse()
	previous_train_ckpt = None
	create_initial_experiment_dir(opts)
	coach = Net(opts, previous_train_ckpt)
	coach.train()

"""
python train_facecnet.py \
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
