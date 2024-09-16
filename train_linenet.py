import os
import random
import torch
from model.linenet import Net
import json
import pprint
from options.train_options import TrainOptions

random.seed(0)
torch.manual_seed(0)

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

''''
python train_linenet.py \
--dataset_type linenet_data \
--exp_dir experiment/ \
--val_interval 20000 \
--batch_size 4 \
--max_steps 200000
'''
