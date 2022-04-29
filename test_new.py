import os
import sys
import shutil
import json
import time
import pprint

import torch
import numpy as np
from imantics import Polygons, Mask
import cv2
from tqdm import tqdm

import config as cfg
import inference as inf

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_NUM

TESTSET = 'private_test'
DATA_DIR = cfg.TEST_DIR
IMG_DIR = DATA_DIR + 'images/'
TGT_DIR = DATA_DIR + 'segment_label/'

MODEL_TYPE = cfg.MODEL_TYPE
RUN_NO = cfg.RUN_NO
NUM_CLASSES = cfg.NUM_CLASSES

IOU_THR = 0.85
OUT_DIR = cfg.OUT_DIR
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

anno_file = f"segm_annotations_{MODEL_TYPE}.json"
result_file = f"result_{MODEL_TYPE}_{RUN_NO}_{TESTSET}.json"
stats_file = f"stats_{MODEL_TYPE}_{RUN_NO}_{IOU_THR}.txt"


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

def show_corners(img, corners):
    for corner in corners:
        cv2.circle(img, tuple(map(int, corner)), radius=5, color=(255, 0, 0), thickness=-1)

def create_annos():
    id = 0
    images, annotations = [], []

    for img_file in os.listdir(IMG_DIR):
        img_ext = img_file.split('.')[1]
        if not img_ext in ['jpg', 'png', 'jpeg']:
            print(f"{img_file} is not an image, ignoring.")
            continue
            
        img_name = img_file.split('.')[0]
        tgt_path = TGT_DIR + f"{img_name}.json"
        if not os.path.exists(tgt_path):
            print(f"Can't find label file for [{img_file}]")
            continue
        target = json.load(open(tgt_path))
        image = {
            'id': id,
            'file_name': img_file,
            'height': target['imageHeight'],
            'width': target['imageWidth']
        }

        shapes = [shape for shape in target['shapes'] if shape['label'] != 'con_dau']
        if len(shapes) != 1:
            print(f"No valid annotation or too many annotations for [{img_file}]")
            continue
        
        anno = shapes[0]
        pts_flt = anno['points']
        pts_int = [list(map(int, pt)) for pt in pts_flt]
        segm = []
        for pt in pts_flt:
            segm.extend([pt[0]])
            segm.extend([pt[1]])
        
        BBox = Polygons(pts_flt).bbox()
        bbox = BBox.bbox(style=BBox.WIDTH_HEIGHT)
        w, h = BBox.size
        cls = 1 if MODEL_TYPE == 'single' else cls_map[anno['label']][0]

        annotation = {
            'id': id,
            'image_id': id,
            'iscrowd': 0,
            'category_id': cls,
            'bbox': bbox,
            'area': w * h * 1.,
            'segmentation': [segm],
            'points': pts_int
        }
        id += 1
        
        images.append(image)
        annotations.append(annotation)
    
    # Save anno
    with open(DATA_DIR + anno_file, 'w') as f:
        json.dump({
            'info': {
                'description': "eKYC Segmentation Private Test",
                'version': "v1.0",
                'date_created': "12/04/2022",
                'contributor': "phatdp, cuonghv"
            },
            'categories': [
                {'id': 1, 'name': "CitizenCardV1_back"},
                {'id': 2, 'name': "CitizenCardV1_front"},
                {'id': 3, 'name': "CitizenCardV2_back"},
                {'id': 4, 'name': "CitizenCardV2_front"},
                {'id': 5, 'name': "IdentificationCard_back"},
                {'id': 6, 'name': "IdentificationCard_front"},
                {'id': 7, 'name': "LicenseCard"},
                {'id': 8, 'name': "Other"},
                {'id': 9, 'name': "Passport"}
            ],
            'images': images,
            'annotations': annotations
        }, f, indent=4)

def scoring(checkpoint):
    coco = json.load(open(DATA_DIR + anno_file))
    images = coco['images']
    annos = coco['annotations']
    cats = coco['categories']

    results = {}
        
    for id in tqdm(range(len(images))):
        img_file = images[id]['file_name']
        img_name = img_file.split('.')[0]
        img = cv2.imread(IMG_DIR + img_file)
        
        pts = annos[id]['points']
        gt_pts = np.array(pts)
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
            
            show_corners(vis, gt_pts)
            cv2.putText(vis, f"IoU: {iou:.2f}", (img.shape[1]+30, img.shape[0]-3), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            if not os.path.exists(OUT_DIR + 'output/'):
                os.mkdir(OUT_DIR + 'output/')
            cv2.imwrite(OUT_DIR + 'output/' + img_name + '.png', vis)
        else:
            iou = -1
            if not os.path.exists(OUT_DIR + 'fn_cases/'):
                os.mkdir(OUT_DIR + 'fn_cases/')
            cv2.imwrite(OUT_DIR + 'fn_cases/' + img_name + '.png', img)
        
        if MODEL_TYPE == 'single':
            results.update({
                id: {
                    'img_name': img_name,
                    'iou_score': iou,
                }
            })
        if MODEL_TYPE == 'multi':
            cls_id = annos[id]['category_id']
            target = cats[cls_id-1]['name']
            if preds:
                pred = preds['clss'][0]
                conf = preds['scores'][0]
            else:
                pred = 'No predictions'
                conf = -1.
            
            results.update({
                id: {
                    'img_name': img_name,
                    'iou_score': iou,
                    'target': target,
                    'pred': pred,
                    'conf': float(conf)
                }
            })
    
    with open(OUT_DIR + result_file, 'w+') as g:
        json.dump(results, g, indent=4)

def get_stats(get_fp=True):
    results = json.load(open(OUT_DIR + result_file, 'r'))
    
    test_cases = 0
    result = {}
    fail_cases = []  # clfn fail cases
    if get_fp:
        if not os.path.exists(OUT_DIR + 'fp_cases/'):
            os.mkdir(OUT_DIR + 'fp_cases/')  # segm fp failed cases
        else:
            shutil.rmtree(OUT_DIR + 'fp_cases/')
            os.mkdir(OUT_DIR + 'fp_cases/')
    
    if MODEL_TYPE == 'single':
        tp, fp, fn = 0, 0, 0
    
    for k, v in results.items():
        test_cases += 1
        iou = v['iou_score']
        
        if MODEL_TYPE == 'single':
            if iou == -1:
                fn += 1
            else:
                if iou < IOU_THR:
                    fp += 1
                    if get_fp:
                        shutil.copy(OUT_DIR + f"output/{v['img_name']}.png", OUT_DIR + "fp_cases/")
                elif iou >= IOU_THR:
                    tp += 1
        
        if MODEL_TYPE == 'multi':
            target = v['target']
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
            else:
                if iou < IOU_THR:
                    result[target]['fp'] += 1
                    if get_fp:
                        shutil.copy(OUT_DIR + f"output/{v['img_name']}.png", OUT_DIR + "fp_cases/")
                elif iou >= IOU_THR:
                    result[target]['tp'] += 1
            
            pred = v['pred']
            if pred != target:
                fail_cases.append({
                    'img': v['img_name'],
                    'targ': target,
                    'pred': pred,
                    'conf': v['conf']
                })
            else:
                result[target]['acc'] += 1
    
    # Save result
    with open(OUT_DIR + stats_file, 'w') as g:
        if MODEL_TYPE == 'multi':
            targets = list(sorted(result.keys()))
            all_pre, all_rec, segm_acc, clfn_acc = 0, 0, 0, 0
            g.write(f"{'Class':>25} | {'Cases':>10} | {'P @'f'{IOU_THR}':>10} | {'R @'f'{IOU_THR}':>10} | "
                    + f"{'Acc @'f'{IOU_THR}':>10} | {'Clfn_acc':>10}\n")
            for target in targets:
                num_cases = result[target]['num_cases']
                tp, fp, fn = result[target]['tp'], result[target]['fp'], result[target]['fn']
                pre = tp / (tp + fp)
                rec = tp / (tp + fn)
                acc = result[target]['acc']
                all_pre += pre
                all_rec += rec
                segm_acc += tp
                clfn_acc += acc
                g.write(f"{target:>25} | {num_cases:>10} | {pre:>10.3f} | {rec:>10.3f} | "
                        + f"{tp*100/num_cases:>9.2f}% | {acc*100/num_cases:>9.2f}%\n")
            ap = all_pre / len(targets)
            ar = all_rec / len(targets)
            g.write(f"{'All':>25} | {test_cases:>10} | {ap:>10.3f} | {ar:>10.3f} | "
                    + f"{segm_acc*100/test_cases:>9.2f}% | {clfn_acc*100/test_cases:>9.2f}%\n")
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
            g.write(f"{'Cases':>10} | {'P @'f'{IOU_THR}':>10} | {'R @'f'{IOU_THR}':>10} | {'Acc @'f'{IOU_THR}':>10}\n")
            g.write(f"{test_cases:>10} | {pre:>10.3f} | {rec:>10.3f} | {acc:>9.2f}%\n")
            g.write('\n')

def main():
    test_time = time.time()
    
    if not os.path.exists(DATA_DIR + anno_file):
        print("Creating annos...")
        create_annos()
    else:
        print(f"Anno file [{anno_file}] already created.")
    
    checkpoint = f"segm_{MODEL_TYPE}_{RUN_NO}_best.pth"
    if os.path.exists(cfg.MODEL_DIR + checkpoint):
        print(f"Got checkpoint [{checkpoint}]")
    
    if not os.path.exists(OUT_DIR + result_file):
        print("Scoring...")
        scoring(checkpoint)
    else:
        print(f"Score file [{result_file}] already created.")
    
    print(f"Getting stats with [IoU={IOU_THR}]...")
    get_stats()
    print(f"Stats saved to [{OUT_DIR + stats_file}]")
    
    # sys.stdout = open(OUT_DIR + result_file, 'a')
    test_time = time.time() - test_time
    test_time = time.strftime('%H:%M:%S', time.gmtime(test_time))
    print(f"Total testing time: {test_time}")
    sys.stdout.close()


if __name__ == '__main__':
    main()
