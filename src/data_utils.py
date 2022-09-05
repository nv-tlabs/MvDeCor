# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import os
import numpy as np

from torch.utils.data import DataLoader

from src.dataset import GeneratorIter, initialize_workers


def load_partnet_label_dataset(
        segment_dataset, opt, include_list, categories, train_files, test_files,
        pre_load=False, k_images=-1, run_id=0, means=None, stds=None):
    def collate_fn(x):
        return x

    path = opt.data_dir
    dataset = segment_dataset(
        path,
        load_labels=True,
        max_num_images=opt.max_num_images,
        include_list=include_list,
        k_shots=opt.few_shots,
        categories=categories,
        train_files=train_files,
        test_files=test_files,
        pre_load=pre_load,
        k_images=k_images,
        run_id=run_id,
        means=means,
        stds=stds,
        seg_path=opt.test_data_dir
    )

    segment_train_loader = dataset.load_train(opt.batch_size)
    segment_val_loader = dataset.load_val(opt.test_batch_size)

    segment_train_loader = GeneratorIter(segment_train_loader, int(1e10))
    segment_train_loader = iter(
        DataLoader(
            segment_train_loader,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=initialize_workers,
        )
    )
    segment_val_loader = GeneratorIter(segment_val_loader, int(1e10))
    segment_val_loader = iter(
        DataLoader(
            segment_val_loader,
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
    return dataset, segment_train_loader, segment_val_loader


def load_partnet_selfsup_dataset(
        path, Dataset, logger, opt, include_list, pre_load=False,
        pre_load_size=100, means=None, stds=None):
    import json

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
        pre_load=pre_load,
        pre_load_size=pre_load_size,
        means=means,
        stds=stds
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
    return train_loader, val_loader
