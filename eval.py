import os, sys
import time

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

import config as cfg
from model import get_model
import utils
from engine import evaluate_


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_NUM

TESTSET = 'private_test'
DATA_DIR = cfg.TEST_DIR
IMG_DIR = DATA_DIR + 'images/'

MODEL_TYPE = cfg.MODEL_TYPE
RUN_NO = cfg.RUN_NO
NUM_CLASSES = cfg.NUM_CLASSES

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

anno_file = f"segm_annotations_{MODEL_TYPE}.json"
result_file = f"result_{MODEL_TYPE}_{RUN_NO}_{TESTSET}.txt"

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


def main():
    eval_time = time.time()
    
    checkpoint_path = cfg.MODEL_DIR + f"segm_{MODEL_TYPE}_{RUN_NO}_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"Got checkpoint [segm_{MODEL_TYPE}_{RUN_NO}_best.pth]")
    
    model = get_model(NUM_CLASSES, DEVICE, mode='eval', checkpoint=checkpoint_path)
    
    tfms = T.Compose([T.ToTensor(), T.Normalize(*STATS)])
    ds = CocoDetection(IMG_DIR, DATA_DIR + anno_file, tfms)
    dl = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=utils.collate_fn)
    
    sys.stdout = open(f"result/{result_file}", 'a')
    
    evaluate_(model, dl, DEVICE)
    
    eval_time = time.time() - eval_time
    eval_time = time.strftime('%H:%M:%S', time.gmtime(eval_time))
    print(f"Total testing time: {eval_time}")
    
    sys.stdout.close()


if __name__ == '__main__':
    main()
    