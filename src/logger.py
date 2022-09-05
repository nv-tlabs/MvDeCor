# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import glob
import json
import logging
import os
import shutil
import sys


def setup_logger(name, model_name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
    file_handler = logging.FileHandler(
        "logs_selfsup-proj/logs/{}.log".format(model_name), mode="w"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(handler)
    logger.info("Logger setup finished")
    return logger


def initial_setup(model_name, logger, opt, phase="train"):
    with open(
            "logs_selfsup-proj/configs/{}_{}.json".format(model_name, phase), "w"
    ) as file:
        json.dump(vars(opt), file)
    save_path = "logs_selfsup-proj/scripts/{}_{}/".format(model_name, phase)
    os.makedirs(save_path, exist_ok=True)

    files = glob.iglob("*.py")
    logger.info(files)
    for file in files:
        if os.path.isfile(file):
            logger.info(file)
            shutil.copy(file, "logs_selfsup-proj/scripts/{}_{}/".format(model_name, phase))
    if phase == "train":
        model_save_path = "logs_selfsup-proj/models/{}/".format(model_name)
        os.makedirs(model_save_path, exist_ok=True)


def resume_training(opt):
    from argparse import Namespace
    print(opt)
    with open("logs_selfsup-proj/configs/{}.json".format(opt.pretrained_model)) as file:
        data = json.load(file)
    print("source data: ", data)
    new_opt = Namespace(**data)
    new_opt.init_epoch = opt.init_epoch
    new_opt.model_name = opt.model_name
    print("final data: ", new_opt)
    return new_opt
