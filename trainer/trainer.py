

#-*- coding: utf-8 -*
import os
import sys
import time
import logging
from datetime import datetime


from tools.misc.logger import setlogger
import numpy as np
import numpy.matlib

import torch
from torch import optim
from torch import nn

# from utils.trainer import Trainer
from tools.misc.helper import create_heatmap, Save_Handle, AverageMeter
import models
import datasets
from tools.losses.focal_loss import FocalLoss_BCE_2d


from visdom import Visdom
viz = Visdom(env='keypoints training')


# Add the parent folder to sys.path
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

TARGET_SIZE=128



class ClassTrainer():
    '''
    - Training Class, including loading dataset,
        preprocessing, training, validating, etc.
    '''

    #===================================================================================================================
    #====================================================== Init =======================================================
    #===================================================================================================================

    def __init__(self, args):
        '''
        - Construct ClassTrainer & print the settings
        - Used: Used in train_classifier.py
        :param args: arguments to parse
        '''

        super(ClassTrainer, self).__init__()
        self.args = args
        self.needVisualize = args.visualize

        sub_dir = self.args.model_name + '-' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        self.save_dir = os.path.join(self.args.save_dir, sub_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        setlogger(os.path.join(self.save_dir, 'train.log'))  # set logger

        for k, v in self.args.__dict__.items():  # save args
            logging.info("{}: {}".format(k, v))

    #===================================================================================================================
    #============================================= Intermediate Functions ==============================================
    #===================================================================================================================

    def visualize_(self,epoch, C,input,target,fin1,fin2,fin3):
        '''
        - Visualize the heatmap on Visdom
        - Used: Used in function "train_epoch_"
        :param epoch:
        :param C:
        :param input:
        :param target:
        :param fin1:
        :param fin2:
        :param fin3:
        :return:
        '''

        assert viz.check_connection()

        # Only show the first image in "input"
        for c in range(C):
            viz.heatmap(input[0, 0],opts=dict(colormap='Electric',
                        title='Epoch-{} Input'.format(epoch)))
            viz.heatmap(X=target[0, c],opts=dict(colormap='Electric',
                        title='Epoch-{} Points-{} Target'.format(epoch, c)))
            viz.heatmap(X=fin1[0, c],opts=dict(colormap='Electric',
                        title='Epoch-{} Points-{} Fin1'.format(epoch, c)))
            viz.heatmap(X=fin2[0, c],opts=dict(colormap='Electric',
                        title='Epoch-{} Points-{} Fin2'.format(epoch, c)))
            viz.heatmap(X=fin3[0, c],opts=dict(colormap='Electric',
                        title='Epoch-{} Points-{} Fin3'.format(epoch, c)))
        return


    def train_epoch_(self, epoch, criterion):
        '''
        - Training of the process
        - Used: Used in function "train"
        :param epoch:  Epoch index
        :param criterion:  Training criterion
        :return:
        '''

        # Parameters for print
        step = 0
        step_loss = AverageMeter()
        # step_acc = AverageMeter()
        step_start = time.time()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(epoch)

        epoch_start = time.time()
        self.model.train()  # Set model to training mode
        # self.train_sampler.set_epoch(epoch)

        epoch_loss = AverageMeter()
        # epoch_acc = AverageMeter()

        # Iterate over data.
        #TODO:UNAGUO New added dim: scenes
        # for inputs, labels in self.dataloaders['train']:
        for batch_idx, (inputs, labels) in enumerate(self.dataloaders['train']):
            # inputs = inputs.to(self.device)
            # labels = labels.to(self.device)
            inputs, labels =  inputs.to(self.device), labels.to(self.device)
            # inputs = self.rbm.sample_hidden(inputs)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()
                fin1,fin2,fin3 = self.model(inputs)  #,fin4
                # if labels is not None:
                #     labels = (labels,)
                # out = out.mm(self.crit_mask)
                # labels = labels.mm(self.crit_mask)

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
                loss1 = criterion(fin1.cpu(), target)
                loss2 = criterion(fin2.cpu(), target)
                loss3 = criterion(fin3.cpu(), target)
                # loss4 = criterion(fin4.cpu(), target)

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
                            self.visualize_(epoch, C, inputs.cpu(), target.cpu(), fin1.cpu(), fin2.cpu(), fin3.cpu())
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


    #region Full-Connection
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
    #endregion


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

        # Set Gpu(s)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert self.args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            raise Exception("gpu is not available")


        # Get dataset's Class
        # Tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        dataset_name = getattr(datasets, self.args.dataset_name)
        ### "db_dict"'s format: {'train':Dataset(), 'val':Dataset()}
        self.db_dict = {x: dataset_name(os.path.join(self.args.data_dir,'{}.txt'.format(x)), x) for x in ['train', 'val']}


        # region 3. Load Data with sampler
        # self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.datasets["train"],num_replicas = hvd.size(),rank=hvd.rank())
        # self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.datasets["val"],num_replicas = hvd.size(),rank=hvd.rank())
        # self.dataloaders = {'train':torch.utils.data.DataLoader(
        #                             self.datasets['train'],
        #                             batch_size=args.batch_size,
        #                             #shuffle=True,
        #                             num_workers=24,#args.num_workers
        #                             pin_memory=True,
        #                             sampler=self.train_sampler),
        #                     'val':torch.utils.data.DataLoader(
        #                           self.datasets['val'],
        #                           batch_size=args.batch_size,
        #                           # shuffle=False,
        #                           num_workers=24,#args.num_workers
        #                           pin_memory=True,
        #                           sampler=self.val_sampler)}
        #endregion


        #region 3. Load Data without sampler
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=self.args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=self.args.num_workers, pin_memory=True)
                            for x in ['train', 'val']}
        #endregion


        # Set model as data-parallel mode
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
            self.lr_scheduler = None

        else:
            raise Exception("lr schedule not implement")


        # Resuming previous training by loading weights
        self.start_epoch = 0
        if self.args.resume:
            # self.device = None  #TODO:
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
        self.model.to(self.device)
        # self.model.cuda()


        # self.criterion = BCELogitsLossWithMask()#nn.MSELoss() #nn.CrossEntropyLoss()
        self.criterion = FocalLoss_BCE_2d()#nn.BCELoss()#nn.BCEWithLogitsLoss()# Need to Check reduce=False, size_average=False

        self.crit_mask = torch.FloatTensor([[2.0],[3.0],[3.0],[3.0],[2.0],[2.0],[3.0],[3.0],[3.0],[2.0]]).cuda()
        self.best_dist = 10.0
        self.best_dist_scale = 0.01  #Used in Full-Conv
        self.save_list = Save_Handle(max_num=self.args.max_model_num)
        self.avr_kpdist = np.array([0.0,0.0,0.0,0.0,0.0])
        self.count_kpdist = 0
        self.best_loss = 300.0


    def train(self):
        '''
        - Training & Validating process
        - Used: Used in train_classifier.py
        :return:
        '''

        # self.val_epoch_(0)
        for epoch in range(self.start_epoch, self.args.max_epoch):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, self.args.max_epoch - 1) + '-'*5)
            self.train_epoch_(epoch,self.criterion)

            # if epoch % self.args.val_epoch == 0:
            #     self.val_epoch_(epoch)



# For Testing
if __name__ == '__main__':
    pass