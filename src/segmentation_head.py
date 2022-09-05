# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
from torch import nn


class SegmentationHead2D(torch.nn.Module):
    def __init__(self, in_channel, number_classes, num_layers=1):
        super(SegmentationHead2D, self).__init__()
        convs = []
        bns = []
        module_list = []
        for j in range(num_layers - 1):
            conv = torch.nn.Conv2d(
                in_channel, in_channel, kernel_size=3, padding=True, bias=False
            )
            bn = torch.nn.BatchNorm2d(in_channel)
            convs.append(conv)
            bns.append(bn)
            module_list.append(bn)
            module_list.append(nn.ReLU(inplace=True))
            module_list.append(conv)

        conv = torch.nn.Conv2d(in_channel, number_classes, kernel_size=3, padding=True)
        bn = torch.nn.BatchNorm2d(in_channel)
        module_list.append(bn)
        module_list.append(nn.ReLU(inplace=True))
        module_list.append(conv)
        self.conv = torch.nn.ModuleList(module_list)
        self._init_weight()

    def forward(self, x):
        for c in self.conv:
            x = c(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
