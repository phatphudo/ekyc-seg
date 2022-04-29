"""
Usage:
    python inference.py --data path_to_img/img.png
or
    import inference; result, *_ = inference.run(data='path_to_img/img.png')
"""


import os
import argparse
from re import L

import torch
from torchvision import transforms as T

from PIL import Image
import cv2
import numpy as np
from imantics import Polygons, Mask

import config as cfg
from model import get_model
import utils


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_NUM

MODEL_DIR = 'checkpoints/' # cfg.MODEL_DIR
CWD = os.getcwd()
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CONF = 0.5


def load_img(img_path):
    # img = Image.open(img_path).convert('RGB')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tfms = T.Compose([T.ToTensor(), T.Normalize(*STATS)])
    img = tfms(img)
    return img.to(DEVICE)

def check_img(fname):
    if fname.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        return True
    else:
        return False

def get_latest_model(model_dir, model_type):
    cps = {
        'single': [],
        'multi': []
    }
    
    for cp in list(sorted(os.listdir(model_dir))):
        cp_name = cp.split('.')[0]
        if cp_name.split('_')[-1] == 'best':
            if cp_name.split('_')[1] == 'single':
                cps['single'].append(cp)
            elif cp_name.split('_')[1] == 'multi':
                cps['multi'].append(cp)
            else:
                print(f"Invalid name for checkpoint [{cp}]!")
    
    if len(cps[model_type]) == 0:
        print(f"Found no checkpoint for model [{model_type}]")
        return None
    else:
        return cps[model_type][-1]

def get_pred(pred, confidence=CONF):
    pred_score = list(pred['scores'].detach().cpu().numpy())
    t = []
    for i, x in enumerate(pred_score):
        if x > confidence: 
            t.append(i)
    
    if len(t) == 0:
        return {}
    else:
        pred_t = t[-1]
        # pred_t = np.argmax(np.array(pred_score)) # Take pred
        pred_masks = (pred['masks'] > 0.5).squeeze(0).detach().cpu().numpy()
        pred_class = list(pred['labels'].cpu().numpy())
        # pred_boxes = [[tuple(map(int, (i[0], i[1]))), tuple(map(int, (i[2], i[3])))] for i in list(pred['boxes'].detach().cpu().numpy())]
        pred_boxes = [list(map(int, (i[0], i[1], i[2], i[3]))) for i in list(pred['boxes'].detach().cpu().numpy())]

        return {
            'masks': pred_masks[:pred_t+1],
            'boxes': pred_boxes[:pred_t+1],
            'clss': pred_class[:pred_t+1],
            'scores':pred_score[:pred_t+1]
        }

def segment_img(img_path, preds):
    img = cv2.imread(img_path)
    # print(img.shape)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_org = img.copy()

    # Display detections on img
    for i in range(len(preds['masks'])):
        # print(preds['masks'][i].shape)
        if len(preds['masks'][i].shape) > 2:
            temp = preds['masks'][i].squeeze(0)
            rgb_mask = get_coloured_mask(temp)
        else: 
            rgb_mask = get_coloured_mask(preds['masks'][i])
        # print(rgb_mask.shape)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.35, 0)

        x1, y1, x2, y2 = preds['boxes'][i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
        # TODO: put text for rotated img
        cv2.putText(img, f"{preds['clss'][i]} {preds['scores'][i]:.2f}", (x1-3, y2-3), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Warp perspective detected IDCard
    mask = preds['masks'][0]  # TODO: NMR if 2 or more masks were detected
    if len(mask.shape) > 2:
        mask = mask.squeeze(0)
    polygons = Mask(mask).polygons()
    points = polygons.points
    pts = np.array(points[0])
    warp = perspective(img_org, pts)

    return img, warp

def get_coloured_mask(mask):
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = [128, 0, 128]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def perspective(img, pts):
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    corners = [tl, tr, br, bl]
    src = np.array(corners, dtype='float32')

    # Get the shape of new image
    mins = np.min(pts, axis=0)
    x_min, y_min = map(int, (mins[0], mins[1]))
    maxs = np.max(pts, axis=0)
    x_max, y_max = map(int, (maxs[0], maxs[1]))

    new_w = x_max - x_min
    new_h = y_max - y_min

    # Define destination points on new image
    dst = np.array(
        [[0, 0],
        [new_w - 1, 0],
        [new_w - 1, new_h - 1],
        [0, new_h]],
        dtype='float32')

    # Perform 'reversed' perspective transform
    trans_mat = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, trans_mat, (new_w, new_h))

    if new_w < new_h:
        warp = cv2.rotate(warp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return warp

def combine_imgs(img1, img2, space=30):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    #create empty matrix
    vis = np.zeros((max(h1, h2), w1+w2+space, 3), np.uint8)
    #combine 2 images
    vis[:h1, :w1, :3] = img1
    vis[:h2, w1+space:w1+w2+space, :3] = img2
    return vis


def main(opt):
    img_path = opt.data
    model_type = opt.model
    checkpoint = opt.weights
    inf_mode = opt.mode
    out_dir = opt.output

    # Load img
    img_file = img_path.split('/')[-1]
    if check_img(img_file):
        img = load_img(img_path)
    else:
        print(f"Image file must be in ['jpg', 'png', 'jpeg']!")

    # Load model
    assert model_type in ['single', 'multi']

    if model_type == 'single':
        classes = ['__background__', 'IDCard']
    if model_type == 'multi':
        classes = ['__background__', 'CitizenCardV1_back', 'CitizenCardV1_front', 
            'CitizenCardV2_back', 'CitizenCardV2_front', 'IdentificationCard_back',
            'IdentificationCard_front', 'LicenseCard', 'Other', 'Passport']
    
    num_classes = len(classes)
    
    if checkpoint is None:
        if os.path.exists(MODEL_DIR):
            checkpoint = get_latest_model(MODEL_DIR, model_type)
        else:
            print("No checkpoints directory to load weights!")
    
    model = get_model(num_classes, DEVICE, mode='eval', checkpoint=MODEL_DIR+checkpoint)
    model.eval()

    # Get pred
    pred_dict = model([img])[0]

    if len(pred_dict['labels']) == 0:
        print(f"Model has no predictions for [{img_file}]")
        if inf_mode == 'get_segm':
            return None
        elif inf_mode == 'eval':
            return {}, None
    else:
        # Save result
        assert inf_mode in ['save_pred', 'save_segm', 'get_segm', 'eval']

        preds = get_pred(pred_dict)

        if len(preds.items()) == 0:
            print(f"No predictions satisfying [conf={CONF}] for [{img_file}]")
            if inf_mode == 'get_segm':
                return None
            elif inf_mode == 'eval':
                return preds, None
        else:
            preds['clss'] = [classes[i] for i in preds['clss']]
            
            detected, warp = segment_img(img_path, preds)

            if inf_mode == 'save_segm':
                cv2.imwrite(out_dir + img_file, warp)
                print(f"Output saved at {out_dir}!")
            elif inf_mode == 'save_pred':
                cv2.imwrite(out_dir + img_file, combine_imgs(detected, warp))
                print(f"Output saved at {out_dir}!")
            elif inf_mode == 'get_segm':
                if model_type == 'single':
                    return warp
                else:
                    return preds['clss'][0], warp
            elif inf_mode == 'eval':
                return preds, combine_imgs(detected, warp)


def run(**kwargs):
    # Usage: import inference; inference.run(data='path/img.png')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    return main(opt)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path to image')
    parser.add_argument('--model', type=str, default='single', help='model type (single/multi)')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--mode', type=str, default='save_segm', help='inference mode')
    parser.add_argument('--output', type=str, default='output/')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


# Usage: python inference.py --data path/img.png
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

