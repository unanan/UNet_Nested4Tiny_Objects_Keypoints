#-*- coding: utf-8 -*
from utils.classification_trainer import ClassTrainer
import argparse
import os
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train ')

    # Parameters used in this program
    parser.add_argument('--model-name', default='UNet_Nested', help='the name of the model')
    parser.add_argument('--data-name', default='GearKnob', help='the name of the data')
    parser.add_argument('--device', default='1,2,3', help='assign device')
    parser.add_argument('--resume', default="/home/unaguo/proj/meter-una/gearknob_keypoints/weights/UNet_Nested-0927-091214/best_loss-4.561844646930695_model.pth", help='the path of resume training model')
    parser.add_argument('--batch-size', type=int, default=660, help='input batch size.')
    parser.add_argument('--num-workers', type=int, default=24, help='the num of training process')

    parser.add_argument('--data-dir', default=r'/home/unaguo/proj/meter-una/gearknob_keypoints/datasets/db/gearknob_aug', help='training set directory')
    parser.add_argument('--save-dir', default=r'/home/unaguo/proj/meter-una/gearknob_keypoints/weights', help='directory to save model.')

    parser.add_argument('--optimizer', default='adamw', help='the optimizer method')
    parser.add_argument('--lr', type=float, default=0.000003, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum of gradient')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='the weight decay')#
    parser.add_argument('--lr-scheduler', default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma multiplied at each step')

    parser.add_argument('--steps', default='50, 100, 150, 200, 300, 400', help='the learning rate decay steps')
    parser.add_argument('--display-step', type=int, default=1, help='the num of steps to log training information')
    parser.add_argument('--max-epoch', type=int, default=1000, help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,help='the num of epoch to eval')

    parser.add_argument('--resume-opt', type=bool, default=True, help='whether to load opt state')
    parser.add_argument('--max-model-num', type=int, default=1, help='most recent models num to save ')
    parser.add_argument('--rbm-pretrained', type=bool, default=False, help='whether to pretrain with RBM')


    # Parameters unused in this program
    parser.add_argument('--pretrained', type=bool, default=False, help='whether to load pretrained model')
    parser.add_argument('--fea-dim', type=int, default=512,help='the num of feature dim')
    parser.add_argument('--margin-s', type=float, default=64.0,help='margin s')
    parser.add_argument('--margin-m', type=float, default=0.1,help='margin m')
    parser.add_argument('--embedding-size',type=int, default=128, help='the num of training process')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = ClassTrainer(args)
    trainer.setup()
    trainer.train()
