import os, time
import torch
import torchvision
from torch.utils.data import ConcatDataset, DataLoader, random_split
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.resnet import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.ops import misc as misc_nn_ops
from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models.detection._utils import overwrite_eps

from dataset import SingleDataset, MultiDataset
import tool
from engine import evaluate_


class CardDetector(MaskRCNN):
    def __init__(self, model_type, device, pretrained=False, ft_ext=False):
        
        trainable_backbone_layers = None
        pretrained_backbone = False if pretrained else True
        
        trainable_backbone_layers = _validate_trainable_layers(pretrained or pretrained_backbone, 
                                                               trainable_backbone_layers, 5, 3)
        backbone = resnet50(pretrained=pretrained_backbone, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
        
        super().__init__(backbone, num_classes=91)
        
        if pretrained:
            url = 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
            state_dict = load_state_dict_from_url(url)
            self.load_state_dict(state_dict)
            overwrite_eps(self, 0.0)
        
        assert model_type in ['single', 'multi']
        if model_type == 'single':
            num_classes = 1 + 1
        else:
            num_classes = 1 + 9
        
        if ft_ext:
            for param in self.parameters():
                param.requires_grad = False
        
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_channels = self.roi_heads.mask_predictor.conv5_mask.in_channels
        self.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels, 512, num_classes)
        
        self.to(device)
        
        self.model_type = model_type
        self.device = device
    
    def load_weights(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = checkpoint_path.split('/')[-1]
            state_dict = torch.load(checkpoint_path)
            self.load_state_dict(state_dict, map_location=self.device)
            print(f"Loaded checkpoint [{checkpoint}].")

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        self.model_type = cfg['model']['model_type']
        self.run_no = cfg['model']['run_no']
        self.run_name = f"{self.model_type}_{self.run_no}"

        self.data_dicts = cfg['data']
        self.split_ratio = cfg['dataloader']['split_ratio']
        self.batch_size = cfg['dataloader']['batch_size']
        self.num_workers = cfg['dataloader']['num_workers']
        
        tfms = get_tfms(train=True)
        self.train_dl, self.val_dl = self.get_dl(tfms)
    
        self.model = get_model(cfg)
        
        self.epochs = cfg['trainer']['epochs']
        self.optim_fn = _get_optim(cfg['trainer']['optim_fn'])
        self.optim_hp = cfg['trainer']['optim_hp']
        if cfg['trainer']['lr_scheduler']:
            self.lr_scheduler = _get_sched(cfg['trainer']['lr_scheduler'])
        else:
            self.lr_scheduler = None
        self.writer = SummaryWriter(log_dir=f"runs/{self.run_name}/")
        self.model_dir = cfg['model_dir']
        
        self.test_dir = cfg['test_dir']
        
    def get_dl(self, tfms=None):
        datasets = []
        data_dicts = self.data_dicts
        split_ratio = self.split_ratio
        
        for data_dict in data_dicts:
            data_root = data_dict['data_root']
            img_folder = data_dict['img_folder']
            anno_file = data_dict['anno_file']
            if self.model_type == 'single':
                ds = SingleDataset(
                    os.path.join(data_root, img_folder),
                    os.path.join(data_root, anno_file),
                    tfms
                )
            else:
                ds = MultiDataset(
                    os.path.join(data_root, img_folder),
                    os.path.join(data_root, anno_file),
                    tfms
                )
            datasets.append(ds)
        dataset= ConcatDataset(datasets)
        val_size = int(split_ratio * len(dataset))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, 
                              num_workers=self.num_workers, collate_fn=tool.collate_fn)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, collate_fn=tool.collate_fn)
        
        return train_dl, val_dl
        
    def fit(self):
        torch.cuda.empty_cache()
        
        print("Training...")
        train_time = time.time()
        
        model = self.model
        epochs = self.epochs
        train_dl, val_dl = self.train_dl, self.val_dl
        optimizer = self.optim_fn(self.model.parameters(), **self.optim_hp)
        if self.lr_scheduler is not None:
            lr_sched = self.lr_scheduler(optimizer, self.optim_hp['lr'], epochs=self.epochs, 
                                         steps_per_epoch=len(self.train_dl))
        writer = self.writer
        
        for epoch in range(epochs):
            epoch_time = time.time()

            # Training phase:
            model.train()
            train_losses = []

            for i, batch in enumerate(tqdm(train_dl)):
                loss_dict = self.step(batch)
                loss = sum(loss for loss in loss_dict.values())
                train_losses.append(loss)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if lr_sched is not None:
                    curr_lr = lr_sched.get_last_lr()[0]
                    lr_sched.step()
                else:
                    curr_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar("Iter/learning_rate", curr_lr, epoch*len(train_dl) + i)
                
                for k in sorted(loss_dict.keys()):
                    writer.add_scalar(f"Iter/{k}", loss_dict[k].item(), epoch*len(train_dl) + i)

            if lr_sched is not None:
                last_lr = lr_sched.get_last_lr()[0]
            else:
                last_lr = optimizer.param_groups[0]['lr']

            # Validation phase:
            val_losses = []
            with torch.no_grad():
                for batch in val_dl:
                    loss_dict = self.step(batch)
                    loss = sum(loss for loss in loss_dict.values())
                    val_losses.append(loss)

            train_loss = torch.stack(train_losses).mean().item()
            val_loss = torch.stack(val_losses).mean().item()
            epoch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - epoch_time))
            print(f"Epoch [{epoch:>2d}] : time: {epoch_time} | last_lr: {last_lr:.6f} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
            if (epoch + 1) % 5 == 0:
                print("-"*20)
            
            writer.add_scalars('Epoch/train_loss vs. val_loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            # writer.add_scalar('Epoch/last_lr', last_lr, epoch)

            # Checkpoints
            if epoch == 0:
                best = val_loss
            checkpoint_path = self.model_dir + f'segm_{self.run_name}_last.pth'
            torch.save(model.state_dict(), checkpoint_path)
            if val_loss < best:
                checkpoint_path = self.model_dir + f'segm_{self.run_name}_best.pth'
                torch.save(model.state_dict(), checkpoint_path)
                best = val_loss
        
        train_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_time))
        print(f"Total training time: {train_time}")
        writer.close()
        
    def step(self, batch):
        images, targets = batch
        device = self.device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        return loss_dict

    def eval(self):
        print("Evaluating...")
        eval_time = time.time()
        
        tfms = get_tfms(train=False)
        if self.model_type == 'single':
            ds = SingleDataset(
                os.path.join(self.test_dir, 'images/'),
                os.path.join(self.test_dir, 'annotations.json/'),
                tfms
            )
        else:
            ds = MultiDataset(
                os.path.join(self.test_dir, 'images/'),
                os.path.join(self.test_dir, 'annotations.json/'),
                tfms
            )
        dl = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=tool.collate_fn)
        
        evaluate_(self.model, dl, self.device)
        
        eval_time = time.time() - eval_time
        eval_time = time.strftime('%H:%M:%S', time.gmtime(eval_time))
        print(f"Total testing time: {eval_time}")

def get_model(cfg):
    device = torch.device(cfg['device'])
    model = CardDetector(
        model_type=cfg['model']['model_type'],
        device=device,
        pretrained=cfg['model']['pretrained'],
        ft_ext=cfg['model']['ft_ext']
    )
    return model
    
def get_tfms(train=False):
    stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tfms = []
    tfms.extend([T.ToTensor(),T.Normalize(*stats)])
    if train:
        tfms.extend([T.RandomAutocontrast(),
                     T.ColorJitter(brightness=.5, hue=.2)])
    return T.Compose(tfms)

def _get_optim(optim_name):
    if optim_name == 'adam':
        return torch.optim.Adam
    elif optim_name == 'sgd':
        return torch.optim.SGD
    else:
        raise NotImplementedError

def _get_sched(sched_name):
    if sched_name == 'one_cycle':
        return torch.optim.lr_scheduler.OneCycleLR
    elif sched_name == 'linear':
        return torch.optim.lr_scheduler.LinearLR
    else:
        raise NotImplementedError
