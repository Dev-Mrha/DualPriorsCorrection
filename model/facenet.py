import os
import random

import matplotlib

matplotlib.use('Agg')
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import common, train_utils
import torch.nn.functional as F
from model.unet import U_Net, U_Net_shape
# from model.deformable_conv import U_Net_shape
from criteria.lpips.lpips import LPIPS
from configs import data_configs
from datasets.flow_dataset import FlowDataset
from datasets.EXR_dataset import EXRDataset
from model.tv_loss import TVLoss
from model.psp import pSp
import torchvision.transforms as transforms

random.seed(0)
torch.manual_seed(0)


class Net:
    def __init__(self, opts, prev_train_checkpoint=None):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'
        torch.cuda.set_device(0)
        self.opts.device = self.device
        # Initialize network
        # self.net = model.Model().to(self.device)
        self.net = U_Net_shape().to(self.device)
        state = torch.load(self.opts.load_model, map_location=self.device)
        my_net_dict = self.net.state_dict()
        update_dict = {k:v for k,v in state['state_dict'].items() if k in my_net_dict.keys()} # and "Up" not in k}
        print(update_dict.keys())
        self.net.load_state_dict(update_dict, strict=True)


        self.pSp = pSp(self.opts).to(self.device)
        for param in self.pSp.parameters():
            param.requires_grad=False
        self.tv_loss = TVLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None


    def load_from_train_checkpoint(self, ckpt):
        print('Loading previous training data...')
        self.global_step = ckpt['global_step'] + 1
        self.best_val_loss = ckpt['best_val_loss']
        self.net.load_state_dict(ckpt['state_dict'])

        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f'Resuming training from step {self.global_step}')

    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}
                x, y, y_hat, flow, f = self.forward(batch)
                # print(y.size(), y_hat.size())
                loss, encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, flow, f)
                loss_dict = {**loss_dict, **encoder_loss_dict}
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 500 == 0):
                    self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step != 0 and self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)
                    elif val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                self.global_step += 1

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            cur_loss_dict = {}
            with torch.no_grad():
                x, y, y_hat, flow, f = self.forward(batch)
                loss, cur_encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, flow, f)
                cur_loss_dict = {**cur_loss_dict, **cur_encoder_loss_dict}
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            if batch_idx % 500 == 0:
                self.parse_and_log_images(id_logs, x, y, y_hat,
                                          title='images/test/faces',
                                          subscript='{:04d}'.format(batch_idx))

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        else:
            self.requires_grad(self.net.decoder, False)
        optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset = FlowDataset(source_root=dataset_args['train_source_root'],
                                   target_root=dataset_args['train_target_root'],
                                   flow_root=dataset_args['train_flow_root'],
                                   source_transform=transforms_dict['transform_source'],
                                   target_transform=transforms_dict['transform_test'],
                                   opts=self.opts)
        test_dataset = FlowDataset(source_root=dataset_args['test_source_root'],
                                  target_root=dataset_args['test_target_root'],
                                  flow_root=dataset_args['test_flow_root'],
                                  source_transform=transforms_dict['transform_source'],
                                  target_transform=transforms_dict['transform_test'],
                                  opts=self.opts)
        # train_dataset = EXRDataset(source_root=dataset_args['train_source_root'],
        #                            target_root=dataset_args['train_target_root'],
        #                            flow_root_x=dataset_args['train_flow_x_root'],
        #                            flow_root_y=dataset_args['train_flow_y_root'],
        #                            ldmk_root=dataset_args['train_ldmk_root'],
        #                            source_transform=transforms_dict['transform_test'],
        #                            target_transform=transforms_dict['transform_test'],
        #                            opts=self.opts)
        # test_dataset = EXRDataset(source_root=dataset_args['test_source_root'],
        #                            target_root=dataset_args['test_target_root'],
        #                            flow_root_x=dataset_args['test_flow_x_root'],
        #                            flow_root_y=dataset_args['test_flow_y_root'],
        #                            ldmk_root=dataset_args['test_ldmk_root'],
        #                            source_transform=transforms_dict['transform_test'],
        #                            target_transform=transforms_dict['transform_test'],
        #                            opts=self.opts)

        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, flow, f):
        loss_dict = {}
        id_logs = None
        loss = 0.0

        l1_loss = torch.abs(y_hat - y).mean()
        loss_dict['l1_loss'] = float(l1_loss)
        loss += l1_loss * 2

        tv_loss = self.tv_loss(flow)
        loss_dict['tv_loss'] = float(tv_loss)
        loss += tv_loss * 0.5

        # print(f.shape, flow.shape)
        f_loss = torch.norm(f - flow, p=2, dim=1).mean()
        loss_dict['flow_loss'] = float(f_loss)
        loss += f_loss

        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def tst(self, batch):
        x = batch.to(self.device).float()
        y1_, w = self.pSp(x)
        flow = self.net(x, w)
        return flow

    def forward(self, batch):
        x, y, f = batch
        f = f.to(self.device).float()
        x, y = x.to(self.device).float(), y.to(self.device).float()
        y_1, wlist = self.pSp(x)
        flow = self.net(x, wlist)
        flow = flow.permute(0, 2, 3, 1)
        y_hat = F.grid_sample(x, grid=flow, mode='bilinear', align_corners=True)
        return x, y, y_hat, flow, f

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=1):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i])
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }

        if self.opts.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['global_step'] = self.global_step
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['best_val_loss'] = self.best_val_loss
        return save_dict

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag
