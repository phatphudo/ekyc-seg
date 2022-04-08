import os
import time
from pprint import pprint
from numpy import imag

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

import matplotlib.pyplot as plt

from dataset import eKYCDataset
import model as modellib
# import transforms as T
import utils
from engine import train_one_epoch, evaluate


DATA_DIR = 'data/synthetic/synthesis/'
DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 2
BATCH_SIZE = 2

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
STATS = mean, std
tfms = T.Compose([T.ToTensor(), T.Normalize(*STATS)])

dataset = eKYCDataset(DATA_DIR, DATA_DIR + 'annotations.json', transforms=tfms)

val_size = int(0.1 * len(dataset))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
# print([train_size, val_size])

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True, collate_fn=utils.collate_fn)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE,
                    shuffle=False, collate_fn=utils.collate_fn)

img, target = train_ds[1]
# plt.imshow(img.permute(1, 2, 0))
# plt.show()
# print(target)

model = modellib.get_model(NUM_CLASSES)
model.to(DEVICE)

# images, targets = next(iter(train_dl))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]
# # print(len(images))
# # print(len(targets))
# # pprint(targets)
# # exit()

# output = model(images, targets)
# pprint(output)
# exit()

epochs = 3
lr = 5e-4
optimizer = SGD(model.parameters(), lr)

writer = SummaryWriter()


def end_epoch(epoch, result):
    pass

def fit_one_cycle(model, train_dl, val_dl, epochs, optimizer, scaler=None):
    torch.cuda.empty_cache()

    train_loss = 0.

    for epoch in range(epochs):
        start_time = time.time()

        result = {
            'time': 'abc'
        }

        # Training phase:
        model.train()
        train_losses = []

        for i, batch in enumerate(train_dl):
            images, targets = batch
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()}
                       for t in targets]
            # print(len(images), len(targets))
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)
                loss_sum = sum(loss for loss in loss_dict.values())

            # pprint(loss_dict)

            losses = " | ".join(f"{k}: {loss_dict[k].item():.4f}" for k in sorted(loss_dict.keys()))
            train_loss = loss_sum.item()

            # print(train_loss)
            # exit()

            train_losses.append(loss_sum)

            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()

            # train_loss += train_loss.item()

            writer.add_scalar('training loss',
                            train_loss / 1000,
                            epoch * len(train_dl) + i)

            # evaluate(model, val_dl, DEVICE)

            # keys, values = [], []
            # for k in sorted(loss_dict.keys()):
            #     keys.append(k)
            #     values.append(loss_dict[k].item())
            
            # item_dict = {k: loss_dict[k].item() for k in sorted(loss_dict.keys())}
            # result.update(item_dict)
            
            # pprint(loss_dict)
            # pprint(item_dict)
            # pprint(result)
            # print(losses)
        # losses = ""
        # for k in sorted(loss_dict.keys()):
        #     losses += f"{k}: {loss_dict[k].item()}" + " "

        
        epoch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        print(f"Epoch [{epoch}] : time: {epoch_time} | train_loss: {train_loss:.4f} | " + losses)




# fit_one_cycle(model, train_dl, val_dl, epochs, optimizer)



img, _ = val_ds[0]
# plt.imshow(img.type(torch.int).permute(1, 2, 0))
# plt.show()
# pprint(img)

# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(DEVICE)])
pprint(prediction)

# for epoch in range(epochs):
#     train_one_epoch(model, optimizer=optim_fn, data_loader=train_dl,
#                     device=DEVICE, epoch=epoch, print_freq=5)
#     evaluate(model, data_loader=val_dl, device=DEVICE)
