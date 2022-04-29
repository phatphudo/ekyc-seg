# import torch

# # Config for model 'single'
MODEL_TYPE = 'single'
RUN_NO = 7
NUM_CLASSES = 1 + 1

# # Config for model 'multi'
# MODEL_TYPE = 'multi'
# RUN_NO = 1
# NUM_CLASSES = 1 + 9

## Train hparams
BATCH_SIZE = 2
NUM_WORKERS = 2

# # GPU3
# GPU_NUM = '1'
# DATA_DIR = '/mnt/datadrive2/phatdp/eKYC/v2.1/'
# MODEL_DIR = '/mnt/datadrive2/phatdp/eKYC/checkpoints/'
# DATA_DIR2 = '/mnt/datadrive2/phatdp/eKYC/v2.0/'

# TEST_DIR = '/mnt/datadrive2/phatdp/eKYC/private_dataset_2.0.1/'
# OUT_DIR = '/mnt/datadrive2/phatdp/eKYC/testing/' + f"{MODEL_TYPE}_{RUN_NO}/"

# # GPU2
GPU_NUM = '1'
DATA_DIR = '/mnt/data/phatdp/ekyc/v2.1/'
MODEL_DIR = '/mnt/data/phatdp/ekyc/checkpoints/'
DATA_DIR2 = '/mnt/data/phatdp/ekyc/v2.0/'

TEST_DIR = '/mnt/data/phatdp/ekyc/private_dataset_2.0.1/'
OUT_DIR = '/mnt/data/phatdp/ekyc/testing/' + f"{MODEL_TYPE}_{RUN_NO}/"

