import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import config as cfg


MODEL_TYPE = cfg.MODEL_TYPE
RUN_NO = cfg.RUN_NO
MODEL_DIR = cfg.MODEL_DIR
        

def get_model(num_classes, device, mode='train', pretrained=False, ft_ext=False, checkpoint=None):
    if mode == 'train':
        pretrained = True
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
    if ft_ext:
        for param in model.parameters():
            param.requires_grad = False
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer, num_classes)
    
    if mode == 'resume':
        checkpoint_last = MODEL_DIR + f'segm_{MODEL_TYPE}_{RUN_NO}_last.pth'
        if not checkpoint_last is None:
            model.load_state_dict(torch.load(checkpoint_last, map_location=device))
        else:
            print("No last checkpoint found to load.")

    if mode == 'eval':
        if not checkpoint is None:
            model.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            print("Please provide checkpoint to load!")

    return model.to(device)
