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

    def visualize_(self, epoch, input, target, out_finlist):
        '''
        - Visualize the heatmap on Visdom
        - Used: Used in function "train_epoch_"
        :param epoch:  Epoch index, showed in the heatmaps' title
        :param input:  Input data batch
        :param target: Target heatmap batch
        :param out_finlist: Output heatmap batch list after training
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

            for idx, fin in enumerate(out_finlist):
                fin_ = fin[0]
                viz.heatmap(
                    X=fin_[c],
                    opts=dict(
                        colormap='Electric',
                        title='Epoch-{} Points-{} Fin{}'.format(epoch, c, idx)))
        return


    def train_epoch_(self, epoch):
        '''
        - Training of the process
        - Used: Used in function "train"
        :param epoch:  Epoch index
        :return:
        '''

        # Parameters for print
        step = 0
        step_loss = AverageMeter()
        # step_acc = AverageMeter()
        step_start = time.time()


        epoch_start = time.time()

        # Set model to training mode
        self.model.train()

        epoch_loss = AverageMeter()
        # epoch_acc = AverageMeter()

        # Iterate over data.
        for batch_idx, (inputs, labels) in enumerate(self.dataloaders['train']):
            # inputs = inputs.to(self.device)
            # labels = labels.to(self.device)
            inputs, labels =  inputs.to(self.device), labels.to(self.device)
            # inputs = self.rbm.sample_hidden(inputs)

            # forward
            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)  #,fin4
                # if labels is not None:
                #     labels = (labels,)
                # out = out.mm(self.crit_mask)
                # labels = labels.mm(self.crit_mask)

                if isinstance(outputs,list):
                    logging.info("More than one branch produce several outputs.")

                _,C,H,W = fin1.shape
                # print(fin1.cpu().shape,create_heatmap(labels.cpu(),H,W).shape)
                # print(labels.cpu())
                target = torch.from_numpy(create_heatmap(labels.cpu(),H,W))
                # print(target.shape)
                # print(fin1.cpu().shape)
                # print(np.max(create_heatmap(labels.cpu(),H,W)[0,0]), np.min(create_heatmap(labels.cpu(),H,W)[0,0]))
                # print(np.max(fin1.cpu().detach().numpy()[0,0]),np.min(fin1.cpu().detach().numpy()[0,0]))
                # print(np.max(fin2.cpu().detach().numpy()[0,0]),np.min(fin2.cpu().detach().numpy()[0,0]))
                # print(np.max(fin3.cpu().detach().numpy()[0,0]),np.min(fin3.cpu().detach().numpy()[0,0]))
                loss1 = self.criterion(fin1.cpu(), target)
                loss2 = self.criterion(fin2.cpu(), target)
                loss3 = self.criterion(fin3.cpu(), target)
                # loss4 = self.criterion(fin4.cpu(), target)

                T = 5
                # loss = 0.25*(loss1+loss2+loss3+loss4)
                loss = 1.0*(loss1+loss2+loss3)/3
                # loss = 1.0*(loss1*(4+T) + loss2*(3+T) + loss3*(2+T) + loss4*(1+T))/(4*T+10)
                # if loss <1:
                #     loss= 1.0*(loss1*(1+T) + loss2*(2+T) + loss3*(3+T) + loss4*(4+T))/(4*T+10)
                #loss = 1.0*(2*loss2 +3*loss3+4*loss4)/9
                #if loss<1:
                #    loss = 1.0*(loss1+2*loss2 +3*loss3+4*loss4)/10

                loss.backward()
                self.optimizer.step()

                step_loss.update(loss.item(), inputs.size(0))

                if self.needVisualize:
                    if (epoch % 5==0) and (step%5000==0):#(epoch % 10==0)
                        try:
                            # Show in visdom
                            self.visualize_(epoch, inputs.cpu(), target.cpu(), [fin1.cpu(), fin2.cpu(), fin3.cpu()])
                        except:
                            pass

                # TODO: Need to check
                # step_acc.update(np.mean((torch.max(out,1)[1]== labels.data.long()).detach().cpu().numpy().astype(np.float32))
                #                 , inputs.size(0))

                if step % self.args.display_step == 0:
                    temp_time = time.time()
                    train_elap = temp_time - step_start
                    step_start = temp_time
                    batch_elap = train_elap / self.args.display_step if step != 0 else train_elap
                    samples_per_s = 1.0 * step_loss.get_count() / train_elap
                    logging.info('Train Epoch: {}  [{}/{}({:.0f}%)]  '
                                 'Train Loss: (1)-{:.4f}, (2)-{:.4f}, (3)-{:.4f}, avg-{:.4f}, '
                                 '{:.1f} examples/sec {:.2f} sec/batch'
                                 .format(epoch, batch_idx*len(inputs),len(self.datasets['train']), 100.*batch_idx/len(self.dataloaders['train']),
                                         loss1,loss2,loss3,step_loss.get_avg(), #step_acc.get_avg(),Step Train Acc: {:.4f},  #(4)-{:.4f}, loss4,
                                         samples_per_s, batch_elap))
                    step_loss.reset()
                    # step_acc.reset()

                step += 1

                epoch_loss.update(loss.item(), inputs.size(0))
                # epoch_acc.update(np.mean((preds == labels.data).detach().cpu().numpy().astype(np.float32))
                #                 , inputs.size(0))

        logging.info('Train: Epoch {}, Epoch Loss: {:.4f} , Cost {:.1f} sec'
                     .format(epoch, epoch_loss.get_avg(), #epoch_acc.get_avg(),Acc: {:.4f}
                             time.time() - epoch_start))

        # Update criterion weight mask
        # if self.count_kpdist % 6 == 1:
        #     print(self.avr_kpdist)
        #     self.avr_kpdist /= 10
        #     minval = np.min(self.avr_kpdist)
        #     self.avr_kpdist /= minval
        #     self.count_kpdist = 0
        #     avr_kpdist_ = self.avr_kpdist.tolist()
        #     avr_kpdist_ *= 2
        #     weightarray = np.array(avr_kpdist_,dtype=float)
        #     shape0 = weightarray.shape[0]
        #     weightarray = weightarray.reshape((shape0 ,1))
        #     self.crit_mask=torch.FloatTensor(weightarray).cuda()
        #     self.avr_kpdist = np.array([0.0,0.0,0.0,0.0,0.0])
            # logging.info("New crit weights: {}".format(weightarray))

        model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()

        if epoch_loss.get_avg() < self.best_loss:
            self.best_loss = epoch_loss.get_avg()
            logging.info("save best loss model epoch {}".format(epoch))
            torch.save(model_state_dic, os.path.join(self.save_dir, 'best_loss-{}_model.pth'.format(self.best_loss)))

        # save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(epoch))
        # torch.save({
        #     'epoch': epoch,
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'model_state_dict': model_state_dic
        # }, save_path)
        # self.save_list.append(save_path)


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
            logging.info("save best dist model epoch {}, avg4 distance: {}".format(epoch, avg3))
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

        self.needVisualize = self.args.visualize

        # Set Gpu(s)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert self.args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            raise Exception("gpu is not available")


        # Get dataset's Class   Tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        self.dataset_item = getattr(datasets, self.args.dataset_name)
        self.dataset_dict = {
            "train": self.dataset_item(
                os.path.join(self.args.data_dir, self.args.dataset_name, 'train.txt'),
                "train"),
            "val": self.dataset_item(
                os.path.join(self.args.data_dir, self.args.dataset_name, 'val.txt'),
                "val")} #TODO: Provided a sample of dataset class


        # Load Data
        if self.args.data_sampler:
            # Load Data with sampler
            self.train_sampler = getattr(sampler, self.args.data_sampler)
            self.dataloaders = {
                'train': torch.utils.data.DataLoader(
                    self.datasets['train'],
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    sampler=self.train_sampler,
                    pin_memory=True),
                'val': torch.utils.data.DataLoader(
                    self.datasets['val'],
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    pin_memory=True)}
        else:
            # Load Data without sampler
            self.dataloader_dict = {
                "train": torch.utils.data.DataLoader(
                    self.dataset_dict["train"],
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    shuffle=True,
                    pin_memory=True),
                "val": torch.utils.data.DataLoader(
                    self.dataset_dict["val"],
                    batch_size=self.args.batch_size,
                    num_workers=self.args.num_workers,
                    shuffle=False,
                    pin_memory=True)}


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
            raise Exception("optimizer not implement")


        # Set the learning scheduler
        if self.args.lr_scheduler == 'step':
            steps = [int(step) for step in self.args.steps.strip().split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=self.args.gamma)

        elif self.args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.args.gamma)

        elif self.args.lr_scheduler == 'fix':
            logging.error("not implement the learning scheduler 'fix'")
            self.lr_scheduler = None

        else:
            raise Exception("lr schedule not implement")


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
            logging.error("call method 'setup' before calling 'train'")
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