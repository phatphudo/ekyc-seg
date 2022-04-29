import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
from imantics import Polygons, Mask
import torch
import numpy as np
from tqdm import tqdm


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

TESTSET = 'eKYC_segmentation'
DATA_DIR = '/mnt/datadrive2/phatdp/eKYC/eKYC_segmentation/'
OUT_DIR = '/mnt/datadrive2/phatdp/eKYC/testing/eKYC_segmentation/'

IMG_DIR = DATA_DIR + 'images/'
TGT_DIR = DATA_DIR + 'segment_label/'

IOU_THR = 0.5
STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

MODEL_TYPE = 'single'
RUN_NO = 1
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


import inference as inf
from model import get_model

checkpoint = inf.get_latest_model('checkpoints/', model_type=MODEL_TYPE)
model = get_model(2, DEVICE, mode='eval', checkpoint='checkpoints/'+checkpoint)


from torch.utils.data import Dataset, ConcatDataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms as T

class CocoDetection(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anno = coco.loadAnns(ann_ids)[0]

        target = {}
        target['image_id'] = img_id
        target['boxes'] = anno['bbox']
        target['boxes'][2] += target['boxes'][0]
        target['boxes'][3] += target['boxes'][1]
        target['area'] = anno['area']
        target['iscrowd'] = 0
        target['masks'] = coco.annToMask(anno)
        if MODEL_TYPE == 'single':
            target['labels'] = 1
        elif MODEL_TYPE == 'multi':
            target['labels'] = anno['category_id']

        for k, v in target.items():
            target[k] = torch.as_tensor(np.array([v]))

        path = coco.loadImgs(img_id)[0]['file_name']
        try:
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
        except OSError:
            print("Cannot load : {}".format(path))

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)
    

import utils

tfms = T.Compose([T.ToTensor(), T.Normalize(*STATS)])
train_ds = CocoDetection(DATA_DIR + 'train', DATA_DIR + 'train/train_annotations.json', tfms)
val_ds = CocoDetection(DATA_DIR + 'val', DATA_DIR + 'val/val_annotations.json', tfms)

ds = ConcatDataset([train_ds, val_ds])
print(len(ds))

dl = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=utils.collate_fn)
print(len(dl))


# dl_iter = iter(dl)
# for i in range(len(dl_iter)):
#     image, target = dl_iter.next()
#     break
# print(target)

import time
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

@torch.inference_mode()
def evaluate_(model, dataloader, device=DEVICE):
    cpu_device = torch.device('cpu')
    model.eval()

    iou_types = ['segm']
    coco = get_coco_api_from_dataset(dataloader.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in tqdm(dataloader):
        images = list(img.to(device) for img in images)
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in output.items()}  for output in outputs]

        res = {target['image_id'].item(): output  for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    return coco_evaluator

from engine import evaluate

import sys

sys.stdout = open(f"result/result_{MODEL_TYPE}_{RUN_NO}_{TESTSET}_2.txt", "w")

evaluate(model, dl, DEVICE)

sys.stdout.close()
