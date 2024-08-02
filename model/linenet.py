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
import torch.nn as nn
from model.linenet import LineNet
from criteria.line_loss import *
from model.unet import U_Net_Line as U_Net
from utils.Calculator import Calculator
from configs import data_configs
from datasets.exr_dataset import EXRDataset
from datasets.labeled_dataset import LabeledData
from criteria.line_loss import LineLoss
from criteria.sym_loss import SymLoss
from criteria.LAM_loss import LAMLoss
from criteria.sobel_loss import Sobel_Loss
import time

random.seed(0)
torch.manual_seed(0)


class Net:
    def __init__(self, opts, prev_train_checkpoint=None):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:1'
        torch.cuda.set_device(0)
        self.opts.device = self.device
        
        self.net = U_Net(in_ch=3, out_ch=2).to(self.device)
        ckpt = torch.load(self.opts.ckpt)
        self.net.load_state_dict(ckpt['state_dict'])
        print('load ckpt...')
        
        self.lineLoss = LineLoss(device=self.device)
        self.symloss = SymLoss()
        self.sobelLoss = Sobel_Loss(device=self.device)
        self.l2loss = nn.MSELoss(reduction='mean')
        self.lamloss = LAMLoss(device=self.device)

        # self.B, self.H, self.W = self.opts.batch_size, 384, 512
        self.B, self.H, self.W = 1, 384, 512
        xx = torch.arange(0, self.W).view(1, -1).repeat(self.H, 1)
        yy = torch.arange(0, self.H).view(-1, 1).repeat(1, self.W)
        xx = xx.view(1, 1, self.H, self.W).repeat(self.B, 1, 1, 1)
        yy = yy.view(1, 1, self.H, self.W).repeat(self.B, 1, 1, 1)
        self.grid = torch.cat((xx, yy), dim=1).float()
        self.grid = self.grid.permute(0, 2, 3, 1).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)

        # Initialize discriminator

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
        """training process"""
        for epoch in range(self.opts.max_steps):
            print(" ")
            print('-' * 5 + 'Epoch {}/{}'.format(epoch, self.opts.max_steps - 1) + '-' * 5)
            self.epoch = epoch
            self.train_eopch()
            if epoch % 5 == 0:
                self.validate()

    def train_eopch(self):
        loss_dict = {}
        epoch_start = time.time()
        self.net.train()
        for batch_idx, batch in enumerate(self.train_dataloader):
            inputs, map_x, map_y, gt = batch
            inputs, map_x, map_y, gt = inputs.to(self.device), map_x.to(self.device), map_y.to(self.device), gt.to(
                self.device)
            outputs = self.net(inputs)
            est_map_x, est_map_y = outputs[:, 0, :, :].unsqueeze(1), outputs[:, 1, :, :].unsqueeze(1)
            outputs = outputs.permute(0, 2, 3, 1)
            outputs = outputs + self.grid
            outputs[:, :, :, 0] = 2 * outputs[:, :, :, 0].clone() / (self.W - 1) - 1.0
            outputs[:, :, :, 1] = 2 * outputs[:, :, :, 1].clone() / (self.H - 1) - 1.0
            y_hat = F.grid_sample(inputs, outputs, mode='bilinear', align_corners=True)
            map = torch.cat([map_x, map_y], dim=1)
            map = map.permute(0, 2, 3, 1)
            map = map + self.grid
            map[:, :, :, 0] = 2 * map[:, :, :, 0].clone() / (self.W - 1) - 1.0
            map[:, :, :, 1] = 2 * map[:, :, :, 1].clone() / (self.H - 1) - 1.0
            y_1 = F.grid_sample(inputs, map, mode='bilinear', align_corners=True)

            loss, encoder_loss_dict, id_logs = self.calc_loss(inputs, gt, map_x, map_y, est_map_x, est_map_y)
            loss_dict = {**loss_dict, **encoder_loss_dict}

            # Update the parameters and loss record
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging related
            if batch_idx % 50 == 0:
                self.parse_and_log_images(id_logs, inputs, gt, y_hat, y_1, title='images/train')
            if self.global_step % 300 == 0:
                self.print_metrics(loss_dict, prefix='train')
                self.log_metrics(loss_dict, prefix='train')

            del inputs, map_x, map_y, gt, outputs, y_hat, y_1
            self.global_step += 1

        self.checkpoint_me(loss_dict, is_best=False)

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.train_dataloader):
            cur_loss_dict = {}
            with torch.no_grad():
                inputs, map_x, map_y, gt = batch
                inputs, map_x, map_y, gt = inputs.to(self.device), map_x.to(self.device), map_y.to(self.device), gt.to(
                    self.device)
                outputs = self.net(inputs)
                est_map_x, est_map_y = outputs[:, 0, :, :].unsqueeze(1), outputs[:, 1, :, :].unsqueeze(1)
                outputs = outputs.permute(0, 2, 3, 1)
                outputs = outputs + self.grid
                outputs[:, :, :, 0] = 2 * outputs[:, :, :, 0].clone() / (self.W - 1) - 1.0
                outputs[:, :, :, 1] = 2 * outputs[:, :, :, 1].clone() / (self.H - 1) - 1.0
                y_hat = F.grid_sample(inputs, outputs, mode='bilinear', align_corners=True)
                map = torch.cat([map_x, map_y], dim=1)
                map = map.permute(0, 2, 3, 1)
                map = map + self.grid
                map[:, :, :, 0] = 2 * map[:, :, :, 0].clone() / (self.W - 1) - 1.0
                map[:, :, :, 1] = 2 * map[:, :, :, 1].clone() / (self.H - 1) - 1.0
                y_1 = F.grid_sample(inputs, map, mode='bilinear', align_corners=True)
                loss, cur_encoder_loss_dict, id_logs = self.calc_loss(inputs, gt, map_x, map_y, est_map_x, est_map_y)
                cur_loss_dict = {**cur_loss_dict, **cur_encoder_loss_dict}
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            if batch_idx % 20 == 0:
                self.parse_and_log_images(id_logs, inputs, gt, y_hat, y_1,
                                          title='images/test',
                                          subscript='{:04d}'.format(batch_idx))

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()

        if self.best_val_loss is None or loss < self.best_val_loss:
            self.best_val_loss = loss
            self.checkpoint_me(loss_dict, is_best=True)

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
        train_dataset = LabeledData(dataset_args['train_dir'], 512, 384, down_sample_factor=1)
        test_dataset = LabeledData(dataset_args['test_dir'], 512, 384, down_sample_factor=1)
        '''
        test_dataset = EXRDataset(source_root=dataset_args['test_source_root'],
                                  target_root=dataset_args['test_target_root'],
                                  flow_root_x=dataset_args['test_flow_x_root'],
                                  flow_root_y=dataset_args['test_flow_y_root'],
                                  source_transform=transforms_dict['transform_source'],
                                  target_transform=transforms_dict['transform_test'],
                                  opts=self.opts)
        '''
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, gt, output, map_x, map_y, est_map_x, est_map_y):
        loss_dict = {}
        id_logs = None
        sobel_loss_x = self.sobelLoss(map_x, est_map_x, direction='x')
        sobel_loss_y = self.sobelLoss(map_y, est_map_y, direction='y')
        sobel_loss = (sobel_loss_x + sobel_loss_y) * 5
        loss_dict['sobel_loss_flow'] = sobel_loss.item()
        l2_loss_x = self.l2loss(map_x, est_map_x)
        l2_loss_y = self.l2loss(map_y, est_map_y)
        l2_loss = l2_loss_x + l2_loss_y
        loss_dict['l2_loss_flow'] = l2_loss.item()
        img_loss = self.l2loss(gt, output)
        loss_dict['l2_loss_img'] = img_loss.item()
        sym_loss_x = self.symloss(map_x)
        sym_loss_y = self.symloss(map_y)
        sym_loss = (sym_loss_x + sym_loss_y) * 2
        loss_dict['sym_loss'] = sym_loss.item()
        loss = sobel_loss + l2_loss + img_loss + sym_loss
        loss_dict['loss'] = loss.item()
        return loss, loss_dict, id_logs

    def tst(self, batch):
        x = batch.to(self.device).float()
        flow = self.net(x)
        flow = flow.permute(0, 2, 3, 1)
        outputs = flow + self.grid
        outputs[:, :, :, 0] = 2 * outputs[:, :, :, 0].clone() / (self.W - 1) - 1.0
        outputs[:, :, :, 1] = 2 * outputs[:, :, :, 1].clone() / (self.H - 1) - 1.0
        y = F.grid_sample(x, outputs, mode='bilinear', align_corners=True)
        return flow, y

    def forward(self, x):
        flow = self.net(x)
        return flow

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, y_1, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
                'flow_face': common.tensor2im(y_1[i])
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


if __name__ == '__main__':
    mdl = Net()
