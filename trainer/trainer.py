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
        # step_acc = AverageMeter()
        epoch_loss = AverageMeter()
        # epoch_acc = AverageMeter()


        epoch_start = time.time()


        # Set model to training mode
        self.model.train()

        # Sum of step(s)
        step_sum = len(self.dataloaders['train'])

        # Iterate over data
        for step_idx, (inputs, labels) in enumerate(self.dataloaders['train']):
            step_start = time.time()

            inputs, labels =  inputs.to(self.device), labels.to(self.device)

            # Experimental step of encoding the image before training
            # inputs = self.rbm.sample_hidden(inputs)

            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()

                # Forward
                outputs = self.model(inputs)

                if isinstance(outputs,list):
                    # Produce target heatmap
                    _,_,H,W = outputs[0].shape
                    target = torch.from_numpy(create_heatmap(labels.cpu(),H,W))

                    avgloss=0; losses=[]; outputs_cpu = []
                    for output in outputs:
                        output_ = output.cpu()
                        outputs_cpu.append(output_)

                        loss_ = self.criterion(output_, target)
                        avgloss += loss_
                        losses.append(loss_)

                    avgloss = 1.0* avgloss/len(outputs)
                    avgloss.backward()
                    self.optimizer.step()

                else:
                    # logging.info("One output after forwarding")

                    # Produce target heatmap
                    _, _, H, W = outputs.shape
                    target = torch.from_numpy(create_heatmap(labels.cpu(), H, W))

                    outputs_ = outputs.cpu()
                    outputs_cpu=[outputs_]

                    avgloss = self.criterion(outputs_, target)
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

                # TODO: Need to check
                # step_acc.update(np.mean((torch.max(out,1)[1]== labels.data.long()).detach().cpu().numpy().astype(np.float32))
                #                 , inputs.size(0))

                # Log step training info on terminal
                if step_idx % self.log_step == 0:
                    temp_time = time.time()
                    train_elap = temp_time - step_start
                    step_start = temp_time
                    batch_elap = train_elap / self.log_step if step_idx != 0 else train_elap
                    samples_per_s = 1.0 * step_loss.get_count() / train_elap

                    finish_num = step_idx*len(inputs); sum_num = len(self.dataset_dict['train'])
                    logging.info(
                        'Train Epoch: {}  [{}/{}({:.0f}%)]  '      .format(epoch, finish_num, sum_num, 100.*finish_num/sum_num)+
                        'Train Loss: '+ ('{:.4f}, '*len(losses))   .format(*losses)+
                        'avg-{:.4f}, '                             .format(step_loss.get_avg())+
                        '{:.1f} examples/sec {:.2f} sec/batch'     .format(samples_per_s, batch_elap))
                    step_loss.reset()
                    # step_acc.reset()

                epoch_loss.update(avgloss.item(), inputs.size(0))
                # epoch_acc.update(np.mean((preds == labels.data).detach().cpu().numpy().astype(np.float32))
                #                 , inputs.size(0))

        # Log epoch training info on terminal
        logging.info(
            'Train: Epoch {}, '     .format(epoch)+
            'Epoch Loss: {:.4f} , ' .format(epoch_loss.get_avg())+
            'Cost {:.1f} sec'       .format(time.time() - epoch_start))


        # model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
        #
        # if epoch_loss.get_avg() < self.best_loss:
        #     self.best_loss = epoch_loss.get_avg()
        #     logging.info("Save best loss model epoch {}".format(epoch))
        #     torch.save(model_state_dic, os.path.join(self.save_dir, 'best_loss-{}_model.pth'.format(self.best_loss)))
        #


    def val_epoch_(self, epoch):
        '''
        - Validating of the process
        - Used: Used in function "train"
        :param epoch:  Epoch index
        :return:
        '''

        epoch_start = time.time()
        self.model.eval()

        dist_mat1 = np.array([])
        dist_mat2 = np.array([])
        dist_mat3 = np.array([])
        dist_mat4 = np.array([])
        for inputs, labels in self.dataloaders['val']:
            # inputs = inputs.to(self.device)
            inputs = inputs.cuda()
            with torch.set_grad_enabled(False):
                fin1,fin2,fin3 = self.model(inputs) #,fin4
                N,C,H,W = fin1.shape


                #N,C,H,W
                xy1 = self.cal_pos(fin1.cpu())
                xy2 = self.cal_pos(fin2.cpu())
                xy3 = self.cal_pos(fin3.cpu())
                # xy4 = self.cal_pos(fin4.cpu())
                # print(xy3)
                # print(labels.cpu())

                xy1_disval = self.cal_dist(xy1,labels.cpu())
                xy2_disval = self.cal_dist(xy2,labels.cpu())
                xy3_disval = self.cal_dist(xy3,labels.cpu())
                # xy4_disval = self.cal_dist(xy4,labels.cpu())

                dist_mat1 = np.hstack((dist_mat1, xy1_disval))
                dist_mat2 = np.hstack((dist_mat2, xy2_disval))
                dist_mat3 = np.hstack((dist_mat3, xy3_disval))
                # dist_mat4 = np.hstack((dist_mat4, xy4_disval))

        avg1 = np.sum(dist_mat1) / (dist_mat1.shape[0]+1)
        avg2 = np.sum(dist_mat2) / (dist_mat2.shape[0]+1)
        avg3 = np.sum(dist_mat3) / (dist_mat3.shape[0]+1)
        # avg4 = np.sum(dist_mat3) / (dist_mat3.shape[0]+1)

        logging.info("Avg: (1)-{} (2)-{} (3)-{}".format(avg1,avg2,avg3)) # (4)-{} ,avg4

        model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()

        if avg3 < self.best_dist:
            self.best_dist = avg3
            logging.info("Save best dist model epoch {}, avg4 distance: {}".format(epoch, avg3))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_epoch{}_avg4.pth'.format(epoch)))

        # if avg2 < self.best_dist:
        #     self.best_dist = avg2
        #     logging.info("save best dist model epoch {}, avg2 distance: {}".format(epoch, avg2))
        #     torch.save(model_state_dic, os.path.join(self.save_dir, 'best_epoch{}_avg2.pth'.format(epoch)))
        #
        # if avg1 < self.best_dist:
        #     self.best_dist = avg1
        #     logging.info("save best dist model epoch {}, avg1 distance: {}".format(epoch, avg1))
        #     torch.save(model_state_dic, os.path.join(self.save_dir, 'best_epoch{}_avg1.pth'.format(epoch)))


    def cal_dist_fc(self,outs, labels):
        dis_matrix_ = outs-labels
        dis_matrix_ *=dis_matrix_
        out_dis_matrix = []
        for ele in dis_matrix_:
            dis_ele = []
            for idx in range(5):
                dis_ele.append(np.sqrt(ele[idx]+ele[idx+5]))
            out_dis_matrix.append(dis_ele)
            # logging.info("Dist: {}".format(dis_ele))

        out_dis_matrix = np.mean(out_dis_matrix, axis=0)

        return out_dis_matrix


    def val_epoch_fc(self, epoch):
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        dist = [96,96,96,96,96]

        # Iterate over data.
        for inputs, labels in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            with torch.set_grad_enabled(False):
                outs = self.model(inputs)
                # TODO: Need to check the dimension
                # dist = np.linalg.norm(outs.cpu() - labels.cpu())
                # print(outs.cpu(),labels.cpu())
                dist = self.cal_dist_fc(outs.cpu(),labels.cpu())
                logging.info("Val: Epoch {}, Dist matrix: {}, Cost {:.1f} sec".format(epoch, dist,time.time()-epoch_start))
                # TODO: Need to add image show


        self.avr_kpdist += dist
        self.count_kpdist +=1
        sum_dist = sum(dist)

        # model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
        # if sum_dist < self.best_dist:
        #     self.best_dist = sum_dist
        #     logging.info("save best dist model epoch {}, sum distance: {}".format(epoch,sum_dist))
        #     torch.save(model_state_dic, os.path.join(self.save_dir, 'best_acc_model.pth'))


    def cal_pos(self, mat):
        # mat: N, C, H, W
        if not (len(mat.shape)==4):
            raise ValueError("Mat should be as N x C x H x W")

        N,C,H,W = mat.shape
        reshaped_mat = mat.reshape(N,C,H*W)
        pos = np.argmax(reshaped_mat, axis=2)  #

        x= pos%W
        y = (pos-x)/W

        z = np.stack((x, y), axis=2)

        # print(y.shape)

        return z  #NxCx2


    def cal_dist(self,xy_mat,labels):
        # NC2: NxCx2  (C = 5)
        # labels: Nx5x2
        # dis_matrix_: NxCx2
        dis_matrix_ = xy_mat-labels.numpy()
        dis_matrix_ *= dis_matrix_
        _,C,_=xy_mat.shape

        # print(dis_matrix_)


        # print("Before:{}".format(dis_matrix_.shape))
        dis_matrix_ = np.sum(dis_matrix_,axis=2)  #dis_matrix_.shape = (N,C)
        avg_dis = 1.0*np.sum(dis_matrix_,axis=1)/C #avg_dis.shape = (N,1)
        # print("After:{}".format(avg_dis.shape))

        std_dis = np.std(dis_matrix_,axis=1)  #std_dis.shape = (N,1)

        # print(avg_dis+std_dis)

        return avg_dis+std_dis

    #===================================================================================================================
    #================================================ Public Functions =================================================
    #===================================================================================================================

    def setup(self):
        '''
        - Setup settings
        - Used: Used in train_classifier.py
        :return:
        '''

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
        self.dataset_item = getattr(datasets, self.args.dataset_name)
        self.dataset_dict = {
            "train": self.dataset_item(os.path.join(self.args.data_dir, self.args.dataset_name, 'train.txt')),
            "val": self.dataset_item(os.path.join(self.args.data_dir, self.args.dataset_name, 'val.txt'))} #TODO: Provided a sample of dataset class


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

        # Set the criterion
        self.criterion = FocalLoss_BCE_2d()


        # Set the parameters of saving strategy #TODO:Need to modify
        self.best_dist = 10.0
        self.best_dist_scale = 0.01  #Used in Full-Conv
        self.save_list = Save_Handle(max_num=self.args.max_model_num)
        self.avr_kpdist = np.array([0.0,0.0,0.0,0.0,0.0])
        self.count_kpdist = 0
        self.best_loss = 300.0

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