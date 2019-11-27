# * Test to create trainer based on TrainerBase in trainer/trainer_base.py:

#-*- coding: utf-8 -*


import train
from trainer.trainer_base import TrainerBase
from trainer.trainer import Trainer

def test_create_trainerbase():
    args = train.parse_args()
    trainer = TrainerBase(args)
test_create_trainerbase()


def test_create_trainer():
    args = train.parse_args()
    trainer = Trainer(args)
test_create_trainer()