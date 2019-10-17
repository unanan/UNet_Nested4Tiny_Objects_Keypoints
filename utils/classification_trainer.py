#-*- coding: utf-8 -*
from utils.trainer import Trainer
from utils.helper import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
from torch import nn
import logging
import numpy as np
import numpy.matlib
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import models
import datasets
from .losses import FocalLoss_BCE_2d
from .rbm import RBM
# import torch.nn.functional as F
# import torch.utils.data.distributed
# import horovod.torch as hvd

TARGET_SIZE=128


def produceTargetHeatmap(target, IMAGE_HEIGHT, IMAGE_WIDTH):
    # target: N C 2  (C=5)
    N, C, _ = target.shape
    R_corner = 3
    R_pointer = 3
    target_matrix = np.zeros((N, 4, IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)  # 0.0~1.0

    for n in range(N):

        #region GearKnob
        #channel0,point0
        pt =0;ch=0
        center_x = target[n, pt, 0]
        center_y = target[n, pt, 1]

        Gauss_map = np.sqrt((np.matlib.repmat(np.arange(IMAGE_WIDTH), IMAGE_HEIGHT, 1) -
                             np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)) ** 2 +
                            (np.transpose(np.matlib.repmat(np.arange(IMAGE_HEIGHT), IMAGE_WIDTH, 1)) -
                             np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)) ** 2)

        Gauss_map = np.exp(-0.5 * Gauss_map / R_corner)
        target_matrix[n, ch] = Gauss_map


        #channel1,point1~3
        ch = 1
        for pt in range(1,4): # 角点后三个点
            center_x = target[n, pt, 0]
            center_y = target[n, pt, 1]

            #
            Gauss_map = np.sqrt((np.matlib.repmat(np.arange(IMAGE_WIDTH), IMAGE_HEIGHT, 1) -
                                 np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)) ** 2 +
                                (np.transpose(np.matlib.repmat(np.arange(IMAGE_HEIGHT), IMAGE_WIDTH, 1)) -
                                 np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)) ** 2)
            Gauss_map = np.exp(-0.5 * Gauss_map / R_pointer)
            target_matrix[n, ch] += Gauss_map
        target_matrix[n, ch] = target_matrix[n, ch]/np.max(target_matrix[n, ch])


        # channel2,point4
        pt = 4;ch = 2 #指针头
        center_x = target[n, pt, 0]
        center_y = target[n, pt, 1]

        Gauss_map = np.sqrt((np.matlib.repmat(np.arange(IMAGE_WIDTH), IMAGE_HEIGHT, 1) -
                             np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)) ** 2 +
                            (np.transpose(np.matlib.repmat(np.arange(IMAGE_HEIGHT), IMAGE_WIDTH, 1)) -
                             np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)) ** 2)

        Gauss_map = np.exp(-0.5 * Gauss_map / R_corner)
        target_matrix[n, ch] = Gauss_map


        # channel3,point5~6
        ch=3
        for pt in range(5,C): # 后2个点在一个channel
            center_x = target[n, pt, 0]
            center_y = target[n, pt, 1]

            #
            Gauss_map = np.sqrt((np.matlib.repmat(np.arange(IMAGE_WIDTH), IMAGE_HEIGHT, 1) -
                                 np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)) ** 2 +
                                (np.transpose(np.matlib.repmat(np.arange(IMAGE_HEIGHT), IMAGE_WIDTH, 1)) -
                                 np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)) ** 2)
            Gauss_map = np.exp(-0.5 * Gauss_map / R_pointer)
            target_matrix[n, ch] += Gauss_map
        target_matrix[n, ch] = target_matrix[n, ch]/np.max(target_matrix[n, ch])
        #endregion

        #region Busbar
        # for idx in range(4):
        #     pt =idx; ch=idx
        #     center_x = target[n, pt, 0]
        #     center_y = target[n, pt, 1]
        #
        #     Gauss_map = np.sqrt((np.matlib.repmat(np.arange(IMAGE_WIDTH), IMAGE_HEIGHT, 1) -
        #                          np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)) ** 2 +
        #                         (np.transpose(np.matlib.repmat(np.arange(IMAGE_HEIGHT), IMAGE_WIDTH, 1)) -
        #                          np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)) ** 2)
        #
        #     Gauss_map = np.exp(-0.5 * Gauss_map / R_corner)
        #     target_matrix[n, ch] = Gauss_map
        #endregion

    return target_matrix


class BCELogitsLossWithMask(nn.Module):

    def __init__(self, size_average=True):
        super(BCELogitsLossWithMask, self).__init__()
        self.size_average = size_average



    def forward(self, input, target, mask=None):
        '''
        :param input: Variable of shape (N, C, H, W)  logits 0~1
        :param target:  Variable of shape (N, C, H, W)  0~1 float
        :param mask: Variable of shape (N, C)  0. or 1.  float
        :return:
        '''


        # print(target[0,0,1]) # target: N C 2  input: N C H W
        _,C,_ = target.shape

        if not (C == input.shape[1]):
            raise ValueError("Target channel ({}) must be the same as input channel ({})".format(C, input.shape[1]))

        N, C, H, W = input.shape
        # print(input.shape)
        # target_matrix = np.zeros((N, C, H, W)).astype(np.float32)  #0.0~1.0
        # for n in range(N):
        #     for c in range(C):
        #         target_matrix[n,c,int(target[n,c,1]/TARGET_SIZE),int(target[n,c,0]/TARGET_SIZE)] = 1.0

        target_matrix = self.produceTargetHeatmap(target,H, W)
        # print(input, np.max(target_matrix))
        target_matrix = torch.from_numpy(target_matrix)


        # BCELogitsLossWithMask
        max_val = (-input).clamp(min=0)
        loss =  - input * target_matrix + max_val + ((-max_val).exp() + (-input - max_val).exp()).log() #TODO:Need to check #input
        if self.size_average:
            # w, h = input.shape
            return loss.sum() / (H*W)
        else:
            return loss.sum()


        # neg_abs = - input.abs()
        # loss = input.clamp(min=0) - input * target_matrix + (1 + neg_abs.exp()).log()
        # return loss.mean()



class ClassTrainer(Trainer):
    def setup(self):
        """initial the datasets, model, loss and optimizer"""

        """setting contex"""
        # hvd.init()

        args = self.args
        if torch.cuda.is_available():

            # torch.cuda.set_device(hvd.local_rank())
            # torch.cuda.manual_seed(42)
            # torch.set_num_threads(1)

            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            raise Exception("gpu is not available")

        Dataset = getattr(datasets, args.data_name)
        self.embedding_size = args.embedding_size
        self.datasets = {x: Dataset(os.path.join(args.data_dir,
                                                 '{}.txt'.format(x)), x)
                         for x in ['train', 'val']}

        # self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.datasets["train"],num_replicas = hvd.size(),rank=hvd.rank())
        # self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.datasets["val"],num_replicas = hvd.size(),rank=hvd.rank())

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers, pin_memory=True)
                            for x in ['train', 'val']}

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

        #RBM
        self.VISIBLE_UNITS = 16384
        HIDDEN_UNITS = 2048
        CD_K = 2
        self.rbm = RBM(self.VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=torch.cuda.is_available())


        if self.device_count > 1:
            self.model = getattr(models, args.model_name)()
            self.model = torch.nn.DataParallel(self.model)
        else:
            self.model = getattr(models, args.model_name)()
        # self.model.cuda()

        # Optimizer
        if args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            from utils.adamw import AdamW
            self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgdw':
            from utils.sgdw import SGDW
            self.optimizer = SGDW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adabound':
            from utils.adabound import AdaBound
            self.optimizer = AdaBound(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")
        # hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        # hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        # self.optimizer = hvd.DistributedOptimizer(self.optimizer,named_parameters=self.model.named_parameters())


        # Learning Scheduler
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.strip().split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")


        self.start_epoch = 0
        if args.resume:
            # self.device = None  #TODO:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume)
                if self.device_count > 1:
                    self.model.module.load_state_dict(checkpoint['model_state_dict'], self.device)
                else:
                    self.model.load_state_dict(checkpoint['model_state_dict'], self.device)
                if args.resume_opt:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.start_epoch = checkpoint['epoch'] + 1
            elif suf == 'pth':
                if self.device_count > 1:
                    self.model.module.load_state_dict(torch.load(args.resume, self.device))
                else:
                    self.model.load_state_dict(torch.load(args.resume, self.device))

        self.model.to(self.device)#TODO:
        # self.margin = Margin(self.device, args.margin_s, args.margin_m, Dataset.num_classes)

        #TODO:UNAGUO
        # self.criterion = BCELogitsLossWithMask()#nn.MSELoss() #nn.CrossEntropyLoss()

        self.criterion = FocalLoss_BCE_2d()#nn.BCELoss()#nn.BCEWithLogitsLoss()# Need to Check reduce=False, size_average=False
        self.crit_mask = torch.FloatTensor([[2.0],[3.0],[3.0],[3.0],[2.0],[2.0],[3.0],[3.0],[3.0],[2.0]]).cuda()
        self.best_dist = 10.0
        self.best_dist_scale = 0.01  #Used in Full-Conv
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.avr_kpdist = np.array([0.0,0.0,0.0,0.0,0.0])
        self.count_kpdist = 0
        self.best_loss = 300.0

    def pretrainrbm_epoch(self,epoch,VISIBLE_UNITS):
        epoch_error = 0.0
        for batch_idx, (inputs, _) in enumerate(self.dataloaders['train']):
        # for batch, _ in train_loader:
            inputs = inputs.view(len(inputs), VISIBLE_UNITS)  # flatten input data

            inputs = inputs.cuda()

            batch_error = self.rbm.contrastive_divergence(inputs)

            epoch_error += batch_error

        print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))

        return


    def showinvis(self,epoch, C,input,target,fin1,fin2,fin3):
        from visdom import Visdom
        viz = Visdom(env='symm')
        assert viz.check_connection()

        for c in range(C):
            # print(fin1.cpu()[n,c][np.newaxis, :].shape)
            # viz.image(
            #     fin1.cpu()[0,c][np.newaxis, :],
            #     opts=dict(title='fin1', caption='fin1'),
            # )
            # viz.image(
            #     fin2.cpu()[0,c][np.newaxis, :],
            #     opts=dict(title='fin2', caption='fin2'),
            # )
            # viz.image(
            #     fin3.cpu()[0,c][np.newaxis, :],
            #     opts=dict(title='fin3', caption='fin3'),
            # )

            viz.heatmap(input[0, 0],
                        opts=dict(colormap='Electric', title='Epoch-{} input'.format(epoch)))
            viz.heatmap(X=target[0, c],
                        opts=dict(colormap='Electric', title='Epoch-{} Points-{} target'.format(epoch, c)))
            viz.heatmap(X=fin1[0, c],
                        opts=dict(colormap='Electric', title='Epoch-{} Points-{} fin1'.format(epoch, c)))
            viz.heatmap(X=fin2[0, c],
                        opts=dict(colormap='Electric', title='Epoch-{} Points-{} fin2'.format(epoch, c)))
            viz.heatmap(X=fin3[0, c],
                        opts=dict(colormap='Electric', title='Epoch-{} Points-{} fin3'.format(epoch, c)))
            # viz.heatmap(X=fin4[0, c],
            #             opts=dict(colormap='Electric', title='Epoch-{} Points-{} fin4'.format(epoch, c)))

        return

    def train_epoch(self, epoch, criterion):
        # Parameters for print
        args = self.args
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
                # print(fin1.cpu().shape,produceTargetHeatmap(labels.cpu(),H,W).shape)
                # print(labels.cpu())
                target = torch.from_numpy(produceTargetHeatmap(labels.cpu(),H,W))
                # print(target.shape)
                # print(fin1.cpu().shape)
                # print(np.max(produceTargetHeatmap(labels.cpu(),H,W)[0,0]), np.min(produceTargetHeatmap(labels.cpu(),H,W)[0,0]))
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


                if  (epoch % 5==0) and (step%5000==0):#(epoch % 10==0)
                    try:
                        # Show in visdom
                        self.showinvis(epoch, C,inputs.cpu(), target.cpu(), fin1.cpu(), fin2.cpu(), fin3.cpu())
                    except:
                        pass

                # TODO: Need to check
                # step_acc.update(np.mean((torch.max(out,1)[1]== labels.data.long()).detach().cpu().numpy().astype(np.float32))
                #                 , inputs.size(0))


                if step % args.display_step == 0:
                    temp_time = time.time()
                    train_elap = temp_time - step_start
                    step_start = temp_time
                    batch_elap = train_elap / args.display_step if step != 0 else train_elap
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


    #region Full-Convolution
    def val_epoch(self, epoch):
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


    def train(self):
        """training process"""
        args = self.args

        # self.val_epoch(0)
        if args.rbm_pretrained:
            print('Training RBM...')
            for epoch in range(10):
                self.pretrainrbm_epoch(epoch,self.VISIBLE_UNITS)
            self.rbm.save_weights()

        for epoch in range(self.start_epoch, args.max_epoch):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.train_epoch(epoch,self.criterion)
            # if epoch % args.val_epoch == 0:
            #     self.val_epoch(epoch)
