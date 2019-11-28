# * train.py:
#   main file of the program, concludes "__main__" entry.
# * Config the arguments and run this file.
#
# * Test Status: Not tested
#

#-*- coding: utf-8 -*
from trainer.trainer import Trainer
import argparse
import os

def parse_args():
    '''
    - To parse arguments
    - Used in __main__

    :return: args:  arguments instance
    '''

    parser = argparse.ArgumentParser(description='Keypoints Training')

    # Basic Configs
    parser.add_argument('--model-name',   default='UNet_Nested', help='the name of the model')
    parser.add_argument('--dataset-name', default='Fruits',      help='the name of the dataset')
    parser.add_argument('--device',       default='1,2,3',       help='assign GPU(s) index')
    parser.add_argument('--resume',       default='',            help='the path of model to resume training')
    parser.add_argument('--batch-size',   default=660, type=int, help='input batch size.')
    parser.add_argument('--num-workers',  default=24,  type=int, help='the num of training process')

    # Dataset Configs
    parser.add_argument('--data-dir',     default='./datasets', help='datasets directory')
    parser.add_argument('--save-dir',     default='./weights',  help='directory to save model.')
    parser.add_argument('--data-sampler', default='',           help='data sampler when create DataLoader' )

    # Training Configs
    parser.add_argument('--optimizer',    default='adamw',              help='the optimizer method')
    parser.add_argument('--momentum',     default=0.9,      type=float, help='the momentum of gradient')
    parser.add_argument('--weight-decay', default=1e-4,     type=float, help='the weight decay')#
    parser.add_argument('--gamma',        default=0.1,      type=float, help='gamma multiplied at each step')
    parser.add_argument('--lr',           default=0.000003, type=float, help='the initial learning rate')
    parser.add_argument('--lr-scheduler', default='step',               help='the learning rate schedule')
    parser.add_argument('--steps',        default='50, 100, 200, 300',  help='the learning rate decay steps')

    # Epoch Configs
    parser.add_argument('--log-step',  default=1,    type=int, help='the interval of steps to log training information')
    parser.add_argument('--max-epoch', default=1000, type=int, help='max training epoch')
    parser.add_argument('--val-epoch', default=1,    type=int, help='the interval of epoch to eval')

    # Visualize Configs
    parser.add_argument('--visualize', default=False, type=bool, help='use Visdom check training heatmap or not')
    parser.add_argument('--vis-epoch', default=10,    type=int,  help='the interval of epoches to visualize')


    # Deprecated Configs
    # parser.add_argument('--resume-opt', type=bool, default=True, help='whether to load opt state')
    # parser.add_argument('--max-model-num', type=int, default=1, help='most recent models num to save ')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    # Set visible gpu(s)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()

    # Start training
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()
