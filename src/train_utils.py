# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import torch
import torch.nn as nn

def self_sup_train_step(selfsupnet, train_loader, optimizer, SelfsupLoss, opt, lamb):
    optimizer.zero_grad()
    data = next(train_loader)[0]
    output = selfsupnet(torch.from_numpy(data["inputs"]).cuda())
    torch.cuda.empty_cache()
    loss = SelfsupLoss(output, data["points"], strategy=opt.sampling) * lamb
    loss.backward()
    optimizer.step()
    loss_ = loss.item()
    del data, loss
    return loss_ / lamb


def segment_train_step(net, train_loader, optimizer, SegmentLoss):
    optimizer.zero_grad()
    data = next(train_loader)[0]
    output = net(torch.from_numpy(data["inputs"]).cuda())
    loss = SegmentLoss.forward(output, data)
    loss.backward()
    optimizer.step()
    loss_ = loss.item()
    del data, loss
    return loss_


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Upscaling the input features using bilinear upsampling
        and then applies to convolutional layers
        :param in_channels: input features size
        :param out_channels: output features size
        """
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            self.up,
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class ExtraUpLayers(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(ExtraUpLayers, self).__init__()
        self.up1 = Up(in_channels, in_channels // 2)
        self.up2 = Up(in_channels // 2, output_channels)

    def forward(self, x):
        return self.up2(self.up1(x))
