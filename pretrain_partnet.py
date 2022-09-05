# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import os

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from src.config import OptInit
from src.dataset import Dataset, initialize_workers
from src.dataset import GeneratorIter
from src.logger import initial_setup, setup_logger, resume_training
from src.loss import *
from src.loss import Loss
from third_party.deeplab.network.modeling import deeplabv3plus_resnet50
from src.train_utils import ExtraUpLayers

opt = OptInit()._get_args()
if opt.resume:
    opt = resume_training(opt)

model_name = opt.model_name
logger = setup_logger(__name__, opt.model_name)
initial_setup(model_name, logger, opt, "train")
include_list = []
if opt.include_depth:
    include_list.append("depth")
if opt.include_normal:
    include_list.append("normal")

print(model_name)
MEANS = None
STDS = None

if opt.texture_norm == "chair":
    print("Choosing chair normalization")
    MEANS = {
        "normal": np.array([0.648, 0.6514, 0.654]),
        "depth": np.array([0.11578736, 0.11578736, 0.11578736]),
        "gray": np.array([0.83328339, 0.82187937, 0.80887031]),
    }

    STDS = {
        "normal": np.array([0.2057, 0.2068, 0.2068]),
        "depth": np.array([0.198, 0.198, 0.198]),
        "gray": np.array([0.23114581, 0.24562174, 0.26177624]),
    }
elif opt.texture_norm == "renderpeople":
    print("Choosing renderpeople normalization")
    MEANS = {
        "normal": np.array([0.9033, 0.9053, 0.9033]),
        "depth": np.array([0.030, 0.030, 0.030]),
        "gray": np.array([0.94157035, 0.9374371, 0.93953519]),
    }

    STDS = {
        "normal": np.array([0.0935, 0.0942, 0.0995]),
        "depth": np.array([0.1140, 0.1140, 0.1140]),
        "gray": np.array([0.18824908, 0.19471223, 0.18958232]),
    }
else:
    print("Choosing grayscale normalization")
    MEANS = {
        "normal": np.array([0.84, 0.84, 0.84]),
        "depth": np.array([0.048, 0.048, 0.048]),
        "gray": np.array([0.94, 0.94, 0.94]),
    }

    STDS = {
        "normal": np.array([0.396, 0.38, 0.38]),
        "depth": np.array([0.132, 0.132, 0.132]),
        "gray": np.array([0.138, 0.138, 0.138]),
    }

if not opt.up_layers:
    net = deeplabv3plus_resnet50(
        num_classes=opt.emb_dims,
        pretrained_backbone=opt.load_imagenet,
        up_sample=True,
        in_channels=len(include_list) * 3 + 3,
    ).cuda()
else:
    net = deeplabv3plus_resnet50(
        num_classes=256,
        pretrained_backbone=opt.load_imagenet,
        up_sample=False,
        in_channels=len(include_list) * 3 + 3,
    ).cuda()
    up_net = ExtraUpLayers(256, output_channels=opt.emb_dims).cuda()
    net = torch.nn.Sequential(net, up_net)

if opt.multi_gpus:
    print("Number of devices being used: ", torch.cuda.device_count())

    logger.info("using multi gpus")
    net = torch.nn.DataParallel(net).cuda()

Loss = Loss()
optimizer = Adam(net.parameters(), lr=opt.lr)

if opt.pretrained_model:
    logger.info("loading pre-trained model", opt.pretrained_model)
    pre_trained_model = torch.load(
        "logs_selfsup-proj/models/{}/model.pth".format(opt.pretrained_model)
    )
    net.load_state_dict(pre_trained_model["model_state_dict"])
    optimizer.load_state_dict(pre_trained_model["optimizer_state_dict"])

scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=opt.patience, verbose=True, min_lr=1e-5
)

path = opt.data_dir
with open(
        path + "extra_files/train_test_split/{}".format(opt.train_file_name), "r"
) as file:
    ids = json.load(file)
ids = [d for d in ids if os.path.isdir(path + d)]

train_ids = ids[0: int(len(ids) * 0.9)]
val_ids = ids[int(len(ids) * 0.9):]

logger.info(
    "Len of train set: {}, len of val set: {}".format(len(train_ids), len(val_ids))
)
dataset = Dataset(
    path,
    train_ids=train_ids,
    val_ids=val_ids,
    max_num_images=opt.max_num_images,
    include_list=include_list,
    means=MEANS,
    stds=STDS
)

batch_size = opt.batch_size
train_loader = dataset.load_train(batch_size)
val_loader = dataset.load_val(opt.test_batch_size)


def collate_fn(x):
    return x


train_loader = GeneratorIter(train_loader, int(1e10))
train_loader = iter(
    DataLoader(
        train_loader,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=opt.train_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=initialize_workers,
    )
)

val_loader = GeneratorIter(val_loader, int(1e10))
val_loader = iter(
    DataLoader(
        val_loader,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=opt.test_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=initialize_workers,
    )
)

epochs = opt.epochs
val_loss_min = 1e6
num_iter_per_epochs = 2000
num_iter_test = 200

for e in range(opt.init_epoch, epochs):
    losses = []
    net.train()
    for j in range(num_iter_per_epochs):
        optimizer.zero_grad()
        data = next(train_loader)[0]
        output = net(torch.from_numpy(data["inputs"]).cuda())

        torch.cuda.empty_cache()

        if opt.multi_gpus:
            loss = Loss(output, np.stack(data["points"], 0), opt.sampling)
            loss = torch.mean(loss)
        else:
            loss = compute_loss_batch(output, data["points"], strategy=opt.sampling)
        loss.backward()
        optimizer.step()
        loss = loss.item()
        print("\r {} {}".format(j, loss), end=" ", flush=True)
        losses.append(loss)
        del data
    logger.info("\n Train loss at {} is: {}".format(e, np.mean(losses)))

    net.eval()
    losses = []
    for j in range(num_iter_test):
        data = next(val_loader)[0]
        with torch.no_grad():
            output = net(torch.from_numpy(data["inputs"]).cuda())
        if opt.multi_gpus:
            loss = Loss(output, np.stack(data["points"], 0), opt.sampling)
            loss = torch.mean(loss)
        else:
            loss = compute_loss_batch(
                output, data["points"], opt.sampling, opt.neg_samples
            )
        losses.append(loss.item())
        del data, output, loss
        torch.cuda.empty_cache()
    losses = np.mean(losses)
    logger.info("Val loss at {} is: {}".format(e, losses))

    scheduler.step(losses)

    if val_loss_min > losses:
        logger.info("Saving model at epoch: {}".format(e))
        torch.save(
            {
                "epoch": e,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": losses,
            },
            "logs_selfsup-proj/models/{}/model.pth".format(model_name),
        )
        val_loss_min = losses
