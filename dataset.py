import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import numpy as np

import config as cfg

MODEL_TYPE = cfg.MODEL_TYPE

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
            print(f"Cannot load : {path}")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)
    
    def get_height_and_width(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_info = coco.loadImgs(img_id)[0]
        return img_info['height'], img_info['width']
