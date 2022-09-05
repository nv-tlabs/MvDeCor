# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import socket

import numpy as np
import torch
from torch.optim import Adam, lr_scheduler

from src.config import OptInit
from src.data_utils import *
from src.dataset import SegmentShapenetDataset, \
    Dataset
from src.logger import initial_setup, setup_logger, resume_training
from src.loss import SegmentationLoss
from src.loss import compute_loss_batch
from src.segmentation_head import SegmentationHead2D
from src.test_utils import test_partnet
from src.train_utils import segment_train_step, self_sup_train_step, ExtraUpLayers
from third_party.deeplab.network.modeling import deeplabv3plus_resnet50

hostname = socket.gethostname()
print(hostname)
opt = OptInit()._get_args()
if opt.resume:
    opt = resume_training(opt)

model_name = opt.model_name
logger = setup_logger(__name__, opt.model_name)
initial_setup(model_name, logger, opt, "train")
logger.info(opt)
logger.info(hostname)

print(model_name)
print(opt)

include_list = []
if opt.include_depth:
    include_list.append("depth")
if opt.include_normal:
    include_list.append("normal")

categories = {'Bed': 15,
              'Bottle': 9,
              'Chair': 39,
              'Clock': 11,
              'Dishwasher': 7,
              'Display': 4,
              'Door': 5,
              'Earphone': 10,
              'Faucet': 12,
              'Knife': 10,
              'Lamp': 41,
              'Microwave': 6,
              'Refrigerator': 7,
              'StorageFurniture': 24,
              'Table': 51,
              'TrashCan': 11,
              'Vase': 6}
max_labels = categories[opt.category]

categories = [opt.category]

if not opt.up_layers:
    net_ = deeplabv3plus_resnet50(
        num_classes=opt.emb_dims,
        pretrained_backbone=opt.load_imagenet,
        up_sample=True,
        in_channels=len(include_list) * 3 + 3,
    ).cuda()
    if opt.freeze_bb:
        for params in net_.backbone.parameters():
            params.requires_grad = False
else:
    net_ = deeplabv3plus_resnet50(
        num_classes=256,
        pretrained_backbone=opt.load_imagenet,
        up_sample=False,
        in_channels=len(include_list) * 3 + 3,
    ).cuda()
    if opt.freeze_bb:
        for params in net_.backbone.parameters():
            params.requires_grad = False
    up_net = ExtraUpLayers(256, output_channels=opt.emb_dims).cuda()
    net_ = torch.nn.Sequential(net_, up_net)

if opt.multi_gpus:
    logger.info("using multi gpus")
    net_ = torch.nn.DataParallel(net_)

if opt.pretrained_model:
    logger.info("loading pre-trained model", opt.pretrained_model)
    pre_trained_model = torch.load(
        "logs_selfsup-proj/models/{}/model.pth".format(
            opt.pretrained_model)
    )
    net_.load_state_dict(pre_trained_model["model_state_dict"])

seg_net = SegmentationHead2D(
    in_channel=opt.emb_dims, number_classes=max_labels,
    num_layers=opt.up_seg_layers
).cuda()
if opt.multi_gpus:
    logger.info("using multi gpus")
    seg_net = torch.nn.DataParallel(seg_net)

net = torch.nn.Sequential(net_, seg_net)
if opt.pretrained_model:
    net_.eval()
    seg_net.train()
    training_net = seg_net
else:
    training_net = net

optimizer = Adam(net.parameters(), lr=opt.lr)
Loss = SegmentationLoss()

scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma,
                                       last_epoch=-1)
batch_size = opt.batch_size
segment_dataset, segment_train_loader, segment_val_loader = \
    load_partnet_label_dataset(
        segment_dataset=SegmentShapenetDataset,
        opt=opt,
        include_list=include_list,
        categories=categories,
        train_files="train_{}.json".format(categories[0]),
        test_files="test_{}.json".format(categories[0]),
        pre_load=True,
        k_images=opt.few_images,
        run_id=opt.run_id
    )


def collate_fn(x):
    return x


if opt.lamb > 0:
    selfsup_train_loader, selfsup_val_loader = \
        load_partnet_selfsup_dataset(
            opt.selfsup_data_dir,
            Dataset,
            logger,
            opt,
            include_list=include_list,
            pre_load=True,
            pre_load_size=100)

epochs = opt.epochs
val_acc_max = 0
num_iter_per_epochs = 40

for e in range(opt.init_epoch, epochs):
    losses = []
    selfsuplosses = []
    training_net.train()
    for j in range(num_iter_per_epochs):
        segment_loss = segment_train_step(net, segment_train_loader, optimizer,
                                          Loss)
        if opt.lamb > 0:
            self_sup_loss = self_sup_train_step(net_, selfsup_train_loader,
                                                optimizer, compute_loss_batch, opt,
                                                opt.lamb)
        else:
            self_sup_loss = 0.0

        print("\r {} {} {}".format(j, segment_loss, self_sup_loss), end=" ", flush=True)
        losses.append(segment_loss)
        selfsuplosses.append(self_sup_loss)
        torch.cuda.empty_cache()

    logger.info("\n Train loss at {} is: {}".format(e, np.mean(losses)))
    logger.info(
        "\n Train Self-sup loss at {} is: {}".format(e, np.mean(selfsuplosses)))
    if e % 50 == 0 and e > 0:
        training_net.eval()
        losses = []
        precisions = []
        class_avgs = []
        for j in range(50):
            optimizer.zero_grad()
            data = next(segment_val_loader)[0]
            with torch.no_grad():
                output = net(torch.from_numpy(data["inputs"]).cuda())
                loss = Loss.forward(output, data)
                precision = Loss.accuracy_batch(output, data,
                                                max_labels=max_labels)
            losses.append(loss.item())
            precisions.append(precision[0])
            class_avgs.append(precision[1])

        losses = np.mean(losses)
        logger.info("Val loss at {} is: {}".format(e, losses))
        logger.info("Val precision at {} is: {}".format(e, np.mean(precisions)))
        logger.info("Val class avg at {} is: {}".format(e, np.mean(class_avgs)))

    scheduler.step()

    logger.info("Saving model at epoch: {}".format(e))
    os.makedirs("logs_selfsup-proj/models/{}/".format(model_name),
                exist_ok=True)
    torch.save(
        {
            "epoch": e,
            "model_state_dict": net.state_dict(),
            "loss": losses,
        },
        "logs_selfsup-proj/models/{}/model.pth".format(model_name),
    )

test_metrics = test_partnet(net, opt, segment_dataset, model_name, voting=opt.voting)
logger.info(test_metrics)
