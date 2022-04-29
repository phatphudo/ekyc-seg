import os
import time
from pprint import pprint
from engine import evaluate, evaluate_
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import config as cfg
import utils
from model import get_model
from dataset import CocoDetection


# GPU3
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_NUM

# Model
MODEL_TYPE = cfg.MODEL_TYPE
RUN_NO = cfg.RUN_NO
NUM_CLASSES = cfg.NUM_CLASSES
MODEL_DIR = cfg.MODEL_DIR
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    print(DEVICE, torch.cuda.get_device_name())

STATS = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
tfms = T.Compose([T.RandomAutocontrast(),
                  T.ColorJitter(brightness=.5, hue=.2), 
                  T.ToTensor(), 
                  T.Normalize(*STATS)])
SPLIT_RATIO = 0.1

# Loader
BATCH_SIZE = cfg.BATCH_SIZE
NUM_WORKERS = cfg.NUM_WORKERS

# Training hyperparameters
epochs = 20
optim_fn = optim.Adam
optim_hp = {
    'lr': 1e-4,
    'weight_decay': 1e-4
}
lr_scheduler = optim.lr_scheduler.OneCycleLR

## Dataset
DATA_DIR = cfg.DATA_DIR
DATA_DIR2 = cfg.DATA_DIR2
print(DATA_DIR)
print(os.listdir(DATA_DIR), len(os.listdir(DATA_DIR + 'images')))

## Evaluate
TEST_DIR = cfg.TEST_DIR
IMG_DIR = TEST_DIR + 'images/'


def step(model, batch):
    images, targets = batch
    images = list(image.to(DEVICE) for image in images)
    targets = [{k: v.to(DEVICE) for k, v in t.items()}
                for t in targets]
    loss_dict = model(images, targets)
    return loss_dict

writer = SummaryWriter(log_dir=f"runs/{MODEL_TYPE}_{RUN_NO}/")

def fit(model, train_dl, val_dl, epochs, optim_fn, optim_hp, lr_scheduler=None):
    torch.cuda.empty_cache()

    train_time = time.time()

    optimizer = optim_fn(model.parameters(), **optim_hp)
    if not lr_scheduler is None:
        lr_sched = lr_scheduler(optimizer, optim_hp['lr'], epochs=epochs, 
                                steps_per_epoch=len(train_dl))

    for epoch in range(epochs):
        epoch_time = time.time()

        # Training phase:
        model.train()
        train_losses = []

        for i, batch in enumerate(tqdm(train_dl)):
            loss_dict = step(model, batch)
            loss = sum(loss for loss in loss_dict.values())
            train_losses.append(loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if lr_sched is not None:
                # curr_lr = lr_sched.get_last_lr()[0]
                lr_sched.step()
            # else:
            #     curr_lr = optimizer.param_groups[0]['lr']
            # writer.add_scalar("Iter/lr", curr_lr, epoch*len(train_dl) + i)
            
            for k in sorted(loss_dict.keys()):
                writer.add_scalar(f"Iter/{k}", loss_dict[k].item(), epoch*len(train_dl) + i)

        if not lr_sched is None:
            last_lr = lr_sched.get_last_lr()[0]
        else:
            last_lr = optimizer.param_groups[0]['lr']

        # Validation phase:
        val_losses = []
        with torch.no_grad():
            for batch in val_dl:
                loss_dict = step(model, batch)
                loss = sum(loss for loss in loss_dict.values())
                val_losses.append(loss)
        
        # with HiddenPrints():
        #     metrics = get_metrics(model, val_dl, DEVICE)

        train_loss = torch.stack(train_losses).mean().item()
        val_loss = torch.stack(val_losses).mean().item()
        epoch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_time))
        print(f"Epoch [{epoch:>2d}] : time: {epoch_time} | last_lr: {last_lr:.6f} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
        # + f" | acc: {metrics['acc']:.4f} | AP_bbox: {metrics['bbox']:.4f} | AP_segm: {metrics['segm']:.4f}")
        if (epoch + 1) % 5 == 0:
            print("-"*20)
        
        writer.add_scalars('Epoch/train_loss vs. val_loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
        writer.add_scalar('Epoch/last_lr', last_lr, epoch)

        # Checkpoints
        if epoch == 0:
            best = val_loss
        checkpoint_path = MODEL_DIR + f'segm_{MODEL_TYPE}_{RUN_NO}_last.pth'
        torch.save(model.state_dict(), checkpoint_path)
        if val_loss < best:
            checkpoint_path = MODEL_DIR + f'segm_{MODEL_TYPE}_{RUN_NO}_best.pth'
            torch.save(model.state_dict(), checkpoint_path)
            best = val_loss
    
    train_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_time))
    print(f"Total training time: {train_time}")
    writer.close()


import sys

def main():
    torch.manual_seed(42)
    
    ds0 = CocoDetection(DATA_DIR2 + 'images/', DATA_DIR2 + 'annotations.json', transform=tfms)
    ds0_half, _ = random_split(ds0, [int(0.5*len(ds0)), len(ds0) - int(0.5*len(ds0))])
    ds1 = CocoDetection(DATA_DIR + 'images/', DATA_DIR + 'annotations.json', transform=tfms)
    dataset = ds1 + ds0_half

    val_size = int(SPLIT_RATIO * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(train_size, val_size)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=NUM_WORKERS, collate_fn=utils.collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=utils.collate_fn)
    print(len(train_dl), len(val_dl))    
    model = get_model(NUM_CLASSES, DEVICE, pretrained=True)
    # model = CardDetector(MODEL_TYPE, DEVICE, pretrained=True)
    
    sys.stdout = open(f"result/result_{MODEL_TYPE}_{RUN_NO}.txt", 'w')

    print("Training...")
    fit(model, train_dl, val_dl, epochs, optim_fn, optim_hp, lr_scheduler)

    print("Evaluating...")
    eval_time = time.time()
    checkpoint_path = MODEL_DIR + f"segm_{MODEL_TYPE}_{RUN_NO}_best.pth"
    if os.path.exists(checkpoint_path):
        print(f"Got checkpoint [segm_{MODEL_TYPE}_{RUN_NO}_best.pth]")
    model_ = get_model(NUM_CLASSES, DEVICE, mode='eval', checkpoint=checkpoint_path)
    
    anno_file = f"segm_annotations_{MODEL_TYPE}.json"
    tfms_test = T.Compose([T.ToTensor(), T.Normalize(*STATS)])
    ds = CocoDetection(IMG_DIR, TEST_DIR + anno_file, tfms_test)
    dl = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=utils.collate_fn)
    evaluate_(model_, dl, DEVICE)
    eval_time = time.time() - eval_time
    eval_time = time.strftime('%H:%M:%S', time.gmtime(eval_time))
    print(f"Total testing time: {eval_time}")

    sys.stdout.close()

    
if __name__ == '__main__':
    print(f"Config: {MODEL_TYPE} @Run {RUN_NO} on GPU [{cfg.GPU_NUM}].")
    confirm = input("Please confirm (y/n): ")
    if confirm == 'y':
        main()
    elif confirm == 'n':
        sys.exit()
