import os
import random
import matplotlib

matplotlib.use('Agg')
import torch
from model.linenet import Net
import json
import pprint
from argparse import Namespace
from options.train_options import TrainOptions

random.seed(0)
torch.manual_seed(0)


def update_new_configs(ckpt_opts, new_opts):
    for k, v in new_opts.items():
        if k not in ckpt_opts:
            ckpt_opts[k] = v
    if new_opts['update_param_list']:
        for param in new_opts['update_param_list']:
            ckpt_opts[param] = new_opts[param]


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


if __name__ == '__main__':
    opts = TrainOptions().parse()
    previous_train_ckpt = None
    if opts.resume_training_from_ckpt:
        opts, previous_train_ckpt = load_train_checkpoint(opts)
    else:
        create_initial_experiment_dir(opts)

    coach = Net(opts, previous_train_ckpt)
    coach.train()

''''
python train.py \
--dataset_type linenet_data \
--exp_dir experiment/ \
--val_interval 20000 \
--batch_size 4 \
--max_steps 200000

python train.py --dataset_type linenet_data --exp_dir experiment/ --val_interval 20000 --max_steps 200000 --batch_size 2 --save_training_data
'''
