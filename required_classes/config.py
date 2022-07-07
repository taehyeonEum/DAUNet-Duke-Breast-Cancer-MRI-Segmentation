import os
import argparse
import time

def str_to_bool(in_str):
    if in_str == 'True' : return True
    elif in_str == 'False' : return False

def parse_args_(args):

    parser = argparse.ArgumentParser(
        description='Get necessary arguments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--device', type = str, default = 'cuda:0', required=True,
        help = 'select device to run deep learning'
    )
    
    parser.add_argument(
        '--log_param', type = str_to_bool, default = False, required=True,
        help = 'choose to save learning log or not'
    )

    parser.add_argument(
        '--visualization', type = str, choices=['none', '2d', '3d', 'both'], default = 'none', required = True,
        help = 'choose visualization mode : none / 2d / 3d / both'
    )

    now = time.localtime()

    parser.add_argument(
        '--dst', type = str, default = '{:02d}_{:02d}_{:02d}_{:02d}h'.format(now.tm_year-2000, now.tm_mon, now.tm_mday, now.tm_hour),
        help = 'folder that save model parameter and log.txt'
    )

    return vars(parser.parse_known_args(args)[0])

#datapaths
# ENTER local path to "DATA_PATH" where your train and test data are saved
DATA_PATH = '/home/NAS_mount/thum/duke/new_data3'
data_dir_list = os.listdir(DATA_PATH)
TRAIN_IMAGE_PATH = os.path.join(DATA_PATH, 'Train_img')
TRAIN_MASK_PATH  = os.path.join(DATA_PATH, 'Train_msk')
VALID_IMAGE_PATH = os.path.join(DATA_PATH, 'Test_img')
VALID_MASK_PATH  = os.path.join(DATA_PATH, 'Test_msk')

# SHOULD be maken befor run train.py
# OUTPUT_PATH = '/home/NAS_mount/thum/duke/output3'
OUTPUT_PATH = './output'
XL_PATH = os.path.join(OUTPUT_PATH, 'duke_segmentation_log.xlsx')
LOG_MODEL_NUMBER = os.path.join(OUTPUT_PATH, 'latest_model_number.txt')

# train.py will make folders below
MODEL_PATH = os.path.join(OUTPUT_PATH, 'model')
IMAGE_RESULT_PATH = os.path.join(OUTPUT_PATH, 'image_result_2d')
IMAGE_RESULT_PATH_3D = os.path.join(OUTPUT_PATH, 'image_result_3d')
TOTAL_RESULT_PATH = os.path.join(OUTPUT_PATH, 'total_result')
GRAPH_PATH = os.path.join(OUTPUT_PATH, 'graph')
VISUALIAZATION_PATH = os.path.join(OUTPUT_PATH, 'visualization')


# not necessary to change but for record!
# if you want to change setting you have to change in train.py
MODEL_NAME = 'DAUNet'
OPTIMIZER = 'AdamW'
LR_SCHEDULER = 'OneCyclicLR'
LOSS = 'Dice_BCE'

GROWTH_RATE = 15
STARTING_CHANNEL = 10
ENCODING_CHANNELS = [2, 4, 8, 16]
de_channels = []
for i, d in enumerate(ENCODING_CHANNELS):
    if i == 0: num_channel = STARTING_CHANNEL + d * GROWTH_RATE
    else: num_channel = num_channel + d * GROWTH_RATE
    de_channels.append(int(num_channel))
    num_channel /= 2
DECODING_CHANNELS = de_channels[::-1]
# DECODING_CHANNELS = [256, 128, 64, 32]

RANDOM_SEED = 777
RESOLUTION = 256
BATCH_SIZE = 20
NUM_EPOCHS = 12
MAX_LR = 0.005
SMOOTH = 1e-5

POS_WEIGHT = 7
BCE_WEIGHT = 2
BCE2_WEIGHT = 0
DSC_WEIGHT = 10 