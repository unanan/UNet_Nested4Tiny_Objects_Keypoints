# * trainer.py:
#   major codes of training & validating process
# * Inherited from TrainerBase in trainer_base.py
#
# * Test Status: Not tested
#

#-*- coding: utf-8 -*
import os
import sys
import time
import logging
from datetime import datetime
import numpy as np
import numpy.matlib
from visdom import Visdom
viz = Visdom(env='keypoints training')

import torch
from torch import optim
from torch import nn

from trainer.trainer_base import TrainerBase
from tools.misc.heatmap import Heatmap
from tools.misc.helper import create_heatmap, Save_Handle, AverageMeter
from tools.misc import sampler
from tools.losses.focal_loss import FocalLoss_BCE_2d
import models
import datasets

# Add the parent folder to sys.path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

class Trainer(TrainerBase):
    '''
    - Training Class, including loading dataset,
        preprocessing, training, validating, etc.
    '''

    #===================================================================================================================
    #============================================= Intermediate Functions ==============================================
    #===================================================================================================================

    def visualize_(self, epoch, input, target, outputs_list):
        '''
        - Visualize the heatmap on Visdom
        - Used: Used in function "train_epoch_"
        :param epoch:  Epoch index, showed in the heatmaps' title
        :param input:  Input data batch
        :param target: Target heatmap batch
        :param outputs_list: Output heatmap batch list after training
        :return:
        '''

        assert viz.check_connection()

        channel = target.shape[1]
        # Only show the first image, since the data is NxCxHxW
        input_ = input[0, 0]
        target_ = target[0]

        for c in range(channel):
            # Show the same input image several times for easily comparing the results
            viz.heatmap(
                X=input_,
                opts=dict(
                    colormap='Electric',
                    title='Epoch-{} Input'.format(epoch)))

            viz.heatmap(
                X=target_[c],
                opts=dict(
                    colormap='Electric',
                    title='Epoch-{} Points-{} Target'.format(epoch, c)))

            for idx, output in enumerate(outputs_list):
                output_ = output[0]
                viz.heatmap(
                    X=output_[c],
                    opts=dict(
                        colormap='Electric',
                        title='Epoch-{} Points-{} Output-{}'.format(epoch, c, idx)))
        return


    def train_epoch_(self, epoch):
        '''
        - Training of the process
        - Used: Used in function "train"
        :param epoch:  Epoch index
        :return:
        '''

        # Parameters to print
        step_loss = AverageMeter()
        epoch_loss = AverageMeter()
        epoch_start = time.time()

        # Set model to training mode
        self.model.train()

        # Sum of step(s)
        step_sum = len(self.dataloaders['train'])

        # Iterate over data
        for step_idx, (inputs, labels) in enumerate(self.dataloaders['train']):
            step_start = time.time()

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Experimental step of encoding the image before training
            # inputs = self.rbm.sample_hidden(inputs)

            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()

                # Forward
                outputs = self.model(inputs)

                if isinstance(outputs, tuple):
                    # Produce target heatmap
                    _,_,H,W = outputs[0].shape
                    target = torch.from_numpy(create_heatmap(labels.cpu(),H,W))

                    avgloss=0; losses=[]; outputs_cpu = []
                    for output in outputs:
                        output_ = output.cpu()
                        outputs_cpu.append(output_)

                        loss_ = self.heatmap_criterion(output_, target)
                        avgloss += loss_
                        losses.append(loss_)

                    avgloss = 1.0* avgloss/len(outputs)
                    avgloss.backward()
                    self.optimizer.step()

                else:
                    # Produce target heatmap
                    _, _, H, W = outputs.shape
                    target = torch.from_numpy(self.heatmaper.create_heatmap(labels.cpu()))

                    outputs_ = outputs.cpu()
                    outputs_cpu=[outputs_]

                    avgloss = self.heatmap_criterion(outputs_, target)
                    losses = [avgloss]
                    avgloss.backward()
                    self.optimizer.step()

                step_loss.update(avgloss.item(), inputs.size(0))

                # Show in visdom
                if self.need_visualize:
                    if (epoch % self.vis_epoch==0) and (step_idx % step_sum == (step_sum-1)):
                        try:
                            self.visualize_(epoch, inputs.cpu(), target.cpu(), outputs_cpu)
                        except:
                            logging.warning("No visdom server running.")

                # Log step training info on terminal
                if step_idx % self.log_step == 0:
                    temp_time = time.time()
                    train_elap = temp_time - step_start
                    batch_elap = train_elap / self.log_step if step_idx != 0 else train_elap
                    samples_per_s = 1.0 * step_loss.get_count() / train_elap

                    finish_num = step_idx*len(inputs); sum_num = len(self.dataset_dict['train'])
                    logging.info(
                        'Train Epoch: {}  [{}/{}({:.0f}%)]  '      .format(epoch, finish_num, sum_num, 100.*finish_num/sum_num)+
                        'Train Loss: '+ ('{:.4f}, '*len(losses))   .format(*losses)+
                        'avg-{:.4f}, '                             .format(step_loss.get_avg())+
                        '{:.1f} examples/sec {:.2f} sec/batch'     .format(samples_per_s, batch_elap))
                    step_loss.reset()

                epoch_loss.update(avgloss.item(), inputs.size(0))

        # Log epoch training info on terminal
        logging.info(
            'Train: Epoch {}, '     .format(epoch)+
            'Epoch Loss: {:.4f} , ' .format(epoch_loss.get_avg())+
            'Cost {:.1f} sec'       .format(time.time() - epoch_start))

        return


    def val_epoch_(self, epoch):
        '''
        - Validating of the process
        - Used: Used in function "train"
        :param epoch:  Epoch index
        :return:
        '''

        epoch_heatmap_loss = AverageMeter()
        epoch_landmark_loss = AverageMeter()
        epoch_start = time.time()

        # Set model to validating mode
        self.model.eval()


        for inputs, labels in self.dataloaders['val']:
            step_start = time.time()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)

            if isinstance(outputs,tuple):
                target = torch.from_numpy(self.heatmaper.create_heatmap(labels.cpu()))
                heatmap_loss = [];landmark_loss = []
                for output in outputs:
                    # Validate with heatmap loss
                    heatmap_loss.append(self.heatmap_criterion(output.cpu(),target))

                    # Validate with landmark(coordinates) loss
                    preds = self.heatmaper.transfer_points(output.cpu(),labels.cpu())
                    landmark_loss.append(self.landmark_criterion(preds, labels))
                epoch_heatmap_loss.update(heatmap_loss[-1].item(), inputs.size(0))
                epoch_landmark_loss.update(landmark_loss[-1].item(), inputs.size(0))

                # Log step validating info on terminal
                logging.info(
                    'Validate:\t'+
                    'Heatmap Loss: ' +  ("{:.4f} , "*len(heatmap_loss)) .format(*heatmap_loss) +
                    'Landmark Loss: ' + ("{:.4f} , "*len(landmark_loss)).format(*landmark_loss) +
                    'Cost {:.1f} sec'                                   .format(time.time() - step_start))

        # Log epoch validating info on terminal
        logging.info(
            'Validate: Epoch {}, '          .format(epoch) +
            'Heatmap Loss: ' +  "{:.4f} , " .format(epoch_heatmap_loss.get_avg()) +
            'Landmark Loss: ' + "{:.4f} , " .format(epoch_landmark_loss.get_avg()) +
            'Cost {:.1f} sec'               .format(time.time() - epoch_start))

        # Get model state dict
        model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()

        # Weights Saving Strategy
        if self.model_save == "last":
            if epoch_heatmap_loss.get_avg()<self.lastbest_heatmaploss or epoch_landmark_loss.get_avg()<self.lastbest_landmarkloss:
                logging.info("Save best model.")
                torch.save(model_state_dic, os.path.join(
                    self.save_dir,
                    'best_epoch_{}_heatmaploss_{}_landmarkloss_{}.pth'.format(
                        epoch,epoch_heatmap_loss.get_avg(),epoch_landmark_loss.get_avg())))
        else:  # "average"
            pass
            raise NotImplementedError("Weights saving strategy 'average' is not implemented")


    def cal_heatmap_loss(self,):
        pass


    def cal_landmark_loss(self,):
        pass

    #===================================================================================================================
    #================================================ Public Functions =================================================
    #===================================================================================================================

    def setup(self):
        '''
        - Setup settings
        - Used: Used in train_classifier.py
        :return:
        '''

        # Misc Settings
        self.log_step = self.args.log_step

        self.need_visualize = self.args.visualize
        self.vis_epoch = self.args.vis_epoch


        # Set Gpu(s)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('Using {} gpus'.format(self.device_count))
            assert self.args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            raise Exception("GPU is not available")


        # Get dataset's Class   Tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        self.dataset = getattr(datasets, self.args.dataset_name)
        self.dataset_dict = {
            "train": self.dataset(os.path.join(self.args.data_dir, self.args.dataset_name, 'train.txt')),
            "val": self.dataset(os.path.join(self.args.data_dir, self.args.dataset_name, 'val.txt'))} #TODO: Provided a sample of dataset class

        # Set heatmaps' pattern
        self.hm_pattern, w, h = self.dataset_dict["train"].getinfo()
        self.heatmaper = Heatmap(pattern=self.hm_pattern, w=w, h=h)


        # Load Data
        if self.args.data_sampler:
            # Load Data with sampler
            self.train_sampler = getattr(sampler, self.args.data_sampler)
            self.dataloaders = {
                'train': torch.utils.data.DataLoader(
                    self.dataset_dict['train'],
                    batch_size  = self.args.batch_size,
                    num_workers = self.args.num_workers,
                    sampler     = self.train_sampler,
                    pin_memory  = True),
                'val': torch.utils.data.DataLoader(
                    self.dataset_dict['val'],
                    batch_size  = self.args.batch_size,
                    num_workers = self.args.num_workers,
                    shuffle     = False,
                    pin_memory  = True)}
        else:
            # Load Data without sampler
            self.dataloader_dict = {
                "train": torch.utils.data.DataLoader(
                    self.dataset_dict["train"],
                    batch_size  = self.args.batch_size,
                    num_workers = self.args.num_workers,
                    shuffle     = True,
                    pin_memory  = True),
                "val": torch.utils.data.DataLoader(
                    self.dataset_dict["val"],
                    batch_size  = self.args.batch_size,
                    num_workers = self.args.num_workers,
                    shuffle     = False,
                    pin_memory  = True)}


        # Get model & Set model as data-parallel mode
        if self.device_count > 1:
            self.model = getattr(models, self.args.model_name)()
            self.model = torch.nn.DataParallel(self.model)
        else:
            self.model = getattr(models, self.args.model_name)()


        # Set the optimizer
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr          = self.args.lr,
                momentum    = self.args.momentum,
                weight_decay= self.args.weight_decay)

        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr          = self.args.lr,
                weight_decay= self.args.weight_decay)

        elif self.args.optimizer == 'adamw':
            from tools.optimizers.adamw import AdamW
            self.optimizer = AdamW(
                self.model.parameters(),
                lr          = self.args.lr,
                weight_decay= self.args.weight_decay)

        elif self.args.optimizer == 'sgdw':
            from tools.optimizers.sgdw import SGDW
            self.optimizer = SGDW(
                self.model.parameters(),
                lr          = self.args.lr,
                weight_decay= self.args.weight_decay)

        elif self.args.optimizer == 'adabound':
            from tools.optimizers.adabound import AdaBound
            self.optimizer = AdaBound(
                self.model.parameters(),
                lr          = self.args.lr,
                weight_decay= self.args.weight_decay)

        else:
            raise Exception("Optimizer not implement")


        # Set the learning scheduler
        if self.args.lr_scheduler == 'step':
            steps = [int(step) for step in self.args.steps.strip().split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=self.args.gamma)

        elif self.args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.args.gamma)

        elif self.args.lr_scheduler == 'fix':
            logging.error("Not implement the learning scheduler 'fix'")
            self.lr_scheduler = None

        else:
            raise Exception("LR schedule not implement")


        # Resuming previous training by loading weights
        self.start_epoch = 0
        self.model_save = self.args.model_save
        self.max_epoch = self.args.max_epoch
        if self.args.resume:
            suf = self.args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(self.args.resume)
                if self.device_count > 1:
                    self.model.module.load_state_dict(checkpoint['model_state_dict'], self.device)
                else:
                    self.model.load_state_dict(checkpoint['model_state_dict'], self.device)

                if self.args.resume_opt:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.start_epoch = checkpoint['epoch'] + 1

            elif suf == 'pth':
                if self.device_count > 1:
                    self.model.module.load_state_dict(torch.load(self.args.resume, self.device))
                else:
                    self.model.load_state_dict(torch.load(self.args.resume, self.device))


        # Set the model to device(CPU/GPU)
        self.model.to(self.device)  # self.model.cuda()

        # Set the criterions
        self.heatmap_criterion = FocalLoss_BCE_2d()
        self.landmark_criterion = nn.MSELoss()


        # Set the parameters of saving strategy #TODO:Need to modify
        self.lastbest_heatmaploss = 10
        self.lastbest_landmarkloss = 10


        self.SETUPFINISH = True
        return


    def train(self):
        '''
        - Training & Validating process
        - Used: Used in train_classifier.py
        :return:
        '''

        if not self.SETUPFINISH:
            logging.error("Call method 'setup' before calling 'train'")
            return

        for epoch in range(self.start_epoch, self.max_epoch):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, self.max_epoch - 1) + '-'*5)

            # Training
            self.train_epoch_(epoch)

            # Validating
            if epoch % self.args.val_epoch == 0:
                 self.val_epoch_(epoch)

        return