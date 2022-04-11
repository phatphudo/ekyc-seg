import os
import json

import torch
# from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
from PIL import Image


class eKYCDataset(Dataset):
    def __init__(self, root, anno_file, model_type='single', transforms=None):
        self.root = root
        self.coco = json.load(open(anno_file, 'r'))

        coco = self.coco
        imgs, anns, cats = {}, {}, {}

        for img in coco['images']:
            imgs[img['id']] = img
        for ann in coco['annotations']:
            anns[ann['id']] = ann
        for cat in coco['categories']:
            cats[cat['id']] = cat
        
        self.imgs = imgs
        self.ids = list(imgs.keys())
        self.anns = anns
        self.cats = cats

        self.model_type = model_type
        self.transforms = transforms

    def __getitem__(self, index):
        id = self.ids[index]

        img_file = self.imgs[id]['file_name']
        mask_file = self.imgs[id]['mask_name']
        anno = self.anns[id]

        img_path = os.path.join(self.root, 'images', img_file)
        mask_path = os.path.join(self.root, 'masks', mask_file)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        target = {}
        target['image_id'] = torch.tensor([id])
        target['boxes'] = torch.as_tensor([anno['box']], dtype=torch.float32)
        target['area'] = torch.as_tensor([anno['area']])
        target['iscrowd'] = torch.as_tensor([anno['iscrowd']])
        target['masks'] = torch.as_tensor([np.array(mask)], dtype=torch.uint8)
        print(target['masks'].shape)
        exit(-1)
        if self.model_type == 'single':
            target['labels'] = torch.as_tensor([1], dtype=torch.int64)
        elif self.model_type == 'multiple':
            target['labels'] = torch.as_tensor([anno['category_id']], dtype=torch.int64)

        if self.transforms:
            trfm = self.transforms
            img = trfm(img)
        
        return img.type(torch.float32), target

    def __len__(self):
        return len(self.imgs)