import os
import sys
import json
import time

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
from imantics import Polygons, Mask
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

import config as cfg
import inference as inf
from model import get_model
import utils
from engine import evaluate_


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_NUM

TESTSET = 'private_test_aug'
DATA_DIR = 'data/test/private_test/aug/'
IMG_DIR = DATA_DIR + 'images/'

MODEL_TYPE = cfg.MODEL_TYPE
RUN_NO = cfg.RUN_NO
NUM_CLASSES = cfg.NUM_CLASSES

IOU_THR = 0.9
OUT_DIR = 'testing/private_test_aug/' + f"{MODEL_TYPE}_{RUN_NO}/"
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
if not os.path.exists(OUT_DIR + 'fail_cases/'):
    os.mkdir(OUT_DIR + 'fail_cases/')  # segm failed cases
if not os.path.exists(OUT_DIR + 'corr_cases/'):
    os.mkdir(OUT_DIR + 'corr_cases/')  # segm correct cases

anno_file = 'annotations.json'
result_file = f"{OUT_DIR}result_{MODEL_TYPE}_{RUN_NO}_{TESTSET}.txt"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

cls_map = {
    'cccd_1_back': (1, 'CitizenCardV1_back'),
    'cccd_1_front': (2, 'CitizenCardV1_front'),
    'cccd_2_back': (3, 'CitizenCardV2_back'),
    'cccd_2_front': (4, 'CitizenCardV2_front'),
    'cmnd_back': (5, 'IdentificationCard_back'),
    'cmnd_front': (6, 'IdentificationCard_front'),
    'blx_front': (7, 'LicenseCard'),
    'other': (8, 'Other'),
    'passport': (9, 'Passport')
}

def show_cv_img(img):
    plt.figure(figsize=(6, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_corners(img, corners):
    for corner in corners:
        cv2.circle(img, tuple(map(int, corner)), radius=5, color=(255, 0, 0), thickness=-1)

def scoring(checkpoint):
    coco = json.load(open(DATA_DIR + anno_file))
    images = coco['images']
    annos = coco['annotations']
    cats = coco['categories']
    
    test_cases = 0
    result = {}
    fail_cases = []  # clfn fail cases
    if MODEL_TYPE == 'single':
        tp, fp, fn = 0, 0, 0

    for id in tqdm(range(len(images))):
        img_file = images[id]['file_name']
        print(img_file)
        img_name = img_file.split('.')[0]
        print(img_name)
        img = cv2.imread(IMG_DIR + img_file)
        
        pts = annos[id]['segmentation']
        gt_pts = np.array(pts, dtype=np.int32).reshape(-1, 2)
        gt_mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(gt_mask, [gt_pts], [255, 255, 255])
        gt_rs = cv2.bitwise_and(img, gt_mask)

        preds, vis = inf.run(data=IMG_DIR+img_file, mode='eval', model=MODEL_TYPE, weights=checkpoint)
        if preds:
            dt_mask = np.zeros(img.shape, dtype=np.uint8)
            mask = preds['masks'][0]
            if len(mask.shape) > 2:
                mask = mask.squeeze(0)
            dt_points = Mask(mask).polygons().points
            cv2.fillPoly(dt_mask, dt_points, [255, 255, 255])
            dt_rs = cv2.bitwise_and(img, dt_mask)

            i = np.logical_and(gt_rs, dt_rs)
            o = np.logical_or(gt_rs, dt_rs)
            iou = np.sum(i) / np.sum(o)
        else:
            iou = -1

        test_cases += 1
        if MODEL_TYPE == 'single':
            if iou == -1:
                fn += 1
                cv2.imwrite(OUT_DIR + 'fail_cases/' + img_name + '.png', img)
            else:
                show_corners(vis, gt_pts)
                cv2.putText(vis, f"IoU: {iou:.2f}", (img.shape[1]+30, img.shape[0]-3), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                if iou < IOU_THR:
                    fp += 1
                    cv2.imwrite(OUT_DIR + 'fail_cases/' + img_name + '.png', vis)
                elif iou >= IOU_THR:
                    tp += 1
                    cv2.imwrite(OUT_DIR + 'corr_cases/' + img_name + '.png', vis)
        if MODEL_TYPE == 'multi':
            cls_id = annos[id]['category_id']
            target = cats[cls_id-1]['name']
            if not target in result.keys():
                result[target] = {
                    'num_cases': 0,
                    'tp': 0,
                    'fp': 0,
                    'fn': 0,
                    'acc': 0
                }
            result[target]['num_cases'] += 1
            if iou == -1:
                result[target]['fn'] += 1
                cv2.imwrite(OUT_DIR + 'fail_cases/' + img_name + '.png', img)
            else:
                show_corners(vis, gt_pts)
                cv2.putText(vis, f"IoU: {iou:.2f}", (img.shape[1]+30, img.shape[0]-3), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                if iou < IOU_THR:
                    result[target]['fp'] += 1
                    cv2.imwrite(OUT_DIR + 'fail_cases/' + img_name + '.png', vis)
                elif iou >= IOU_THR:
                    result[target]['tp'] += 1
                    cv2.imwrite(OUT_DIR + 'corr_cases/' + img_name + '.png', vis)
            if preds:
                if preds['clss'][0] != target:
                    fail_cases.append({
                        'img': img_file,
                        'targ': target,
                        'pred': preds['clss'][0],
                        'conf': preds['scores'][0]
                    })
                else:
                    result[target]['acc'] += 1
            else:
                fail_cases.append({
                        'img': img_file,
                        'targ': target,
                        'pred': 'No predictions',
                        'conf': -1
                    })
    
    # Save result
    with open(result_file, 'w') as g:
        if MODEL_TYPE == 'multi':
            targets = list(sorted(result.keys()))
            all_pre, all_rec, all_acc = 0, 0, 0
            g.write(f"{'Class':>25} | {'Cases':>10} | {'P @'f'{IOU_THR}':>10} | {'R @'f'{IOU_THR}':>10} | {'Acc':>10}\n")
            for target in targets:
                num_cases = result[target]['num_cases']
                tp, fp, fn = result[target]['tp'], result[target]['fp'], result[target]['fn']
                pre = tp / (tp + fp)
                rec = tp / (tp + fn)
                acc = result[target]['acc']
                all_pre += pre
                all_rec += rec
                all_acc += acc
                g.write(f"{target:>25} | {num_cases:>10} | {pre:>10.3f} | {rec:>10.3f} | {acc*100/num_cases:>9.2f}%\n")
            ap = all_pre / len(targets)
            ar = all_rec / len(targets)
            g.write(f"{'All':>25} | {test_cases:>10} | {ap:>10.3f} | {ar:>10.3f} | {all_acc*100/test_cases:>9.2f}%\n")
            g.write('\n')
            if len(fail_cases) != 0:
                g.write(f"Total failed cases: {len(fail_cases)}\n")
                for case in fail_cases:
                    g.write('-'*50+'\n')
                    g.write(f"Result of image [{case['img']}]:"+'\n')
                    g.write('\t'+f"Target: {case['targ']}"+'\n')
                    g.write('\t'+f"Prediction: {case['pred']}"+'\n')
                    g.write('\t'+f"Confidence: {case['conf']*100:.2f}%"+'\n')
            g.write('\n')
        if MODEL_TYPE == 'single':
            pre = tp / (tp + fp)
            rec = tp / (tp + fn)
            acc = tp * 100 / test_cases
            g.write(f"{'Cases':>10} | {'P @'f'{IOU_THR}':>10} | {'R @'f'{IOU_THR}':>10} | {'Acc':>10}\n")
            g.write(f"{test_cases:>10} | {pre:>10.3f} | {rec:>10.3f} | {acc:>9.2f}%\n")
            g.write('\n')

def main():
    test_time = time.time()
    
    checkpoint = f"segm_{MODEL_TYPE}_{RUN_NO}_best.pth"
    print(checkpoint)
    scoring(checkpoint)
    
    # # Load model
    # if not checkpoint is None:
    #     model = get_model(NUM_CLASSES, DEVICE, mode='eval', checkpoint='checkpoints/'+checkpoint)
    
    # tfms = T.Compose([T.ToTensor(), T.Normalize(*STATS)])
    # ds = CocoDetection(IMG_DIR, DATA_DIR + anno_file, tfms)
    # dl = DataLoader(ds, batch_size=4, num_workers=2, collate_fn=utils.collate_fn)
    
    # sys.stdout = open(result_file, 'a')
    
    # # evaluate_(model, dl, DEVICE)
    
    # test_time = time.time() - test_time
    # test_time = time.strftime('%H:%M:%S', time.gmtime(test_time))
    # print(f"Total testing time: {test_time}")
    
    # sys.stdout.close()


if __name__ == '__main__':
    main()
