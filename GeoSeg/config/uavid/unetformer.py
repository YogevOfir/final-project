"""
UnetFormer for uavid datasets with supervision training
Libo Wang, 2022.02.22
"""
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.uavid_dataset import *
from geoseg.models.UNetFormer import UNetFormer
from tools.utils import Lookahead
from tools.utils import process_model_params
from geoseg.datasets.uavid_dataset_256 import *

# # training hparam
# max_epoch = 31
# ignore_index = 255
# train_batch_size = 8
# val_batch_size = 8
# lr = 6e-4
# weight_decay = 0.01
# backbone_lr = 6e-5
# backbone_weight_decay = 0.01
# num_classes = len(CLASSES)
# classes = CLASSES

# Training hyperparams
max_epoch = 40            # or your choice
ignore_index = 255
train_batch_size = 32       # you can increase since 256×256 is smaller
val_batch_size = 16
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

#weights_name = "unetformer-r18-360-768crop-e20"
weights_name = "255-train_val_no_360-e40"
# weights_name = "unetformer-r18-1024-768crop-e40"
weights_path = "model_weights2/uavid/{}".format(weights_name)
test_weights_name = "last"
log_name = 'uavid/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1

pretrained_ckpt_path = None 
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

# # (Optional) specify your pre-trained 1024 checkpoint to fine-tune from:
# pretrained_ckpt_path = "model_weights2/uavid/unetformer-r18-1024-768crop-e40/unetformer-r18-1024-768crop-e40.ckpt"
# resume_ckpt_path = None


gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning

#  define the network
net = UNetFormer(num_classes=num_classes)
# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)

use_aux_loss = True

# define the dataloader

# train_dataset = UAVIDDataset(data_root='data/uavid/train_val', img_dir='images', mask_dir='masks',
#                              mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=(1024, 1024))

train_dataset = UAVIDDataset(data_root='data/uavid/train_val_no_360', img_dir='images', mask_dir='masks',
                             mode='train', mosaic_ratio=0.0, transform=train_aug_256, img_size=(256, 256))

# val_dataset = UAVIDDataset(data_root='data/uavid/val', img_dir='images', mask_dir='masks', mode='val',
#                           mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))

val_dataset = UAVIDDataset(data_root='data/uavid/val_no_360', img_dir='images', mask_dir='masks', mode='val',
                           mosaic_ratio=0.0, transform=val_aug_256, img_size=(256, 256))


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
#optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

