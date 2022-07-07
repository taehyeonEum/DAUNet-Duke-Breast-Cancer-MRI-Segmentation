import os
import argparse
import time


def str_to_bool(in_str):
    if in_str == 'True' : return True
    elif in_str == 'False' : return False

def parse_args(args):

    parser = argparse.ArgumentParser(
        description='Get necessary arguments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--device', type = str, defualt = 'cuda:0', required=True,
        help = 'select device to run deep learning'
    )
    
    parser.add_argument(
        '--log_param', type = str_to_bool, defualt = False, required=True,
        help = 'choose to save learning log or not'
    )

    parser.add_argument(
        '--gr', type = int, defualt = 15,
        help = 'select growth rate for bottleneck'
    )

    parser.add_argument(
        '--st_channel', type = int, default = 10,
        help = 'choose starting channel to enter Dense procedure'
    )

    parser.add_argument(
        'en_channel', type = int, nargs = '*', default = [2, 4, 8, 16], 
        help = 'compose encoding channels for dense layer'
    )

    parser.add_argument(
        '--random_seed', type = int, default = 777,
        help = 'choose random seed for initialization'
    )

    parser.add_argument(
        '--resol', type = int, default = 256,
        help = 'choose resolution for learning' 
    )

    parser.add_argument(
        '--batch_size', type = int, default = 20,
        help = 'choose batch size for learning'
    )

    parser.add_argument(
        '--epochs', type = int, default = '12',
        help = 'choose number of epochs'
    )

    parser.add_argument(
        '--lr', type = float, default = 0.005,
        help = 'choose moderate learning rate for LR scheduler'
    )

    parser.add_argument(
        '--weights', type = int, default = [2, 4, 10], nargs = 3,
        help = 'compose weights for learning'
    )

    now = time.localtime()

    parser.add_argument(
        '--dst', type = str, default = '{:02d}_{:02d}_{:02d}_{:02d}h'.format(now.tm_year-2000, now.tm_mon, now.tm_mday, now.tm_hour),
        help = 'folder that save model parameter and log.txt'
    )

    return vars(parser.parse_args(args)[0])