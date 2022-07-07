import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as tf
from torch.utils.data import DataLoader

import sys
import os
import stat
import numpy as np
import matplotlib.pyplot as plt
import time
import openpyxl as xl

import required_classes.config as cf
import required_classes.transform as duke_tf
import required_classes.random_seed as rs
import required_classes.dataset as ds
from required_classes.metrics.total_score import total_score
from models.DAUnet import DAUnet
from required_classes.metrics.dsc_score import dice_score
from required_classes.loss_functions.customLoss_dice_bce import customLoss
from required_classes.loss_functions.customLoss_dice_bce3 import customLoss as customLoss3
from required_classes.loss_functions.customLoss_dice_bce3_exp import customLoss as customLoss3_exp
from required_classes.config import parse_args_

from required_classes.train import Train_DBC
from visualization.visualization_2d import visualize_2d
from visualization.visualization_3d import visualize_3d

def main_():
    args = parse_args_(sys.argv[1:])
    model_name = Train_DBC(args)() #you can put model name manualy if you already have trained model

    if args['visualization'] == 'none':
        return
    elif args['visualization'] == '2d':
        visualize_2d(args['device'], model_name)
        return
    elif args['visualization'] == '3d':
        visualize_3d(args['device'], model_name)
        return
    elif args['visualization'] == 'both':
        visualize_2d(args['device'], model_name)
        visualize_3d(args['device'], model_name)
        return

if __name__ == '__main__':
    main_()