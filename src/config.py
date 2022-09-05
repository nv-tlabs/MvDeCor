# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import argparse
import random

import numpy as np
import torch


def convert_str_2_bool(v):
    """
    Converts all sorts of string arguments to appropriate
    bool values.
    :param v: str
    :return: bool value of v
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "yes", "1"):
        return True
    elif v.lower() in ("no", "0", "false"):
        return False
    else:
        assert False, "Only boolean values are expected."


class OptInit:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Self supervision for 3D shapes.")
        parser.add_argument(
            "--phase",
            type=str,
            default="train",
            metavar="N",
            choices=["train", "test"]
        )
        parser.add_argument("--exp_name",
                            type=str,
                            default="",
                            help="experiment name")

        parser.add_argument(
            "--data_dir",
            type=str,
            default="objs/renderings/",
        )

        parser.add_argument(
            "--selfsup_data_dir",
            type=str,
            default="objs/renderings/",
        )

        parser.add_argument(
            "--test_data_dir",
            type=str,
            default="../animals/",
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            default=16,
            metavar="batch_size",
            help="mini-batch size (default:16))",
        )

        parser.add_argument(
            "--epochs",
            type=int,
            default=400,
            metavar="N",
            help="number of episode to train ",
        )

        parser.add_argument(
            "--patience",
            type=int,
            default=4,
            help="patience used in optimizer to decay lr",
        )

        parser.add_argument(
            "--lr",
            type=float,
            default=1e-3,
            metavar="LR",
            help="learning rate (default: 0.001)",
        )

        parser.add_argument(
            "--lamb",
            type=float,
            default=1.0,
            metavar="L",
            help="weighting function for different loss (default: 1.0)",
        )

        parser.add_argument("--multi_gpus", action="store_true", help="use multi-gpus")

        parser.add_argument(
            "--init_epoch",
            type=int,
            default=0,
            help="initial epoch to start with the network",
        )

        parser.add_argument(
            "--run_id",
            type=int,
            default=0,
            help="run_id to be also used as a random seed",
        )

        parser.add_argument(
            "--train_workers",
            type=int,
            default=3,
            help="number of workers used for loading training dataset",
        )

        parser.add_argument(
            "--test_workers",
            type=int,
            default=3,
            help="number of workers used for loading testing dataset",
        )

        parser.add_argument(
            "--max_num_images",
            type=int,
            default=86,
            help="maximum number of images rendered per shape",
        )

        parser.add_argument("--sampling", type=int, default=0, help="sampling strategy")

        parser.add_argument(
            "--gamma",
            type=float,
            default=0.95,
            metavar="G",
            help="gamma learning rate decay rate(default: 0.95)",
        )

        parser.add_argument(
            "--test_batch_size",
            type=int,
            default=2,
            metavar="batch_size",
            help="Test batch size)",
        )

        parser.add_argument(
            "--pretrained_model",
            type=str,
            default="",
            metavar="N",
            help="Pretrained model path",
        )

        parser.add_argument(
            "--viz",
            type=convert_str_2_bool,
            nargs="?",
            const=True,
            default=False,
            help="save viz during testing",
        )

        parser.add_argument(
            "--resume",
            type=convert_str_2_bool,
            nargs="?",
            const=True,
            default=False,
            help="whether to resume training from pre-trained model path",
        )

        parser.add_argument(
            "--up_layers",
            type=convert_str_2_bool,
            nargs="?",
            const=True,
            default=True,
            help="whether to add extra layers",
        )

        parser.add_argument(
            "--few_shots",
            type=int,
            default=10,
            help="number of shapes used to train the model",
        )

        parser.add_argument(
            "--few_images",
            type=int,
            default=-1,
            help="number of images per shape used to supervise the model",
        )

        parser.add_argument(
            "--local_rank",
            type=int,
            default=0,
            help="number of shapes used to train the model",
        )

        parser.add_argument(
            "--all_imgs",
            type=convert_str_2_bool,
            nargs="?",
            const=True,
            default=True,
            help="whether to use all images from a shape for self supervision",
        )

        parser.add_argument(
            "--augment",
            type=convert_str_2_bool,
            nargs="?",
            const=True,
            default=False,
            help="whether to augment",
        )

        parser.add_argument(
            "--up_seg_layers",
            type=int,
            default=1,
            help="number of layers on top of emebddings for segmentation prediction",
        )

        parser.add_argument(
            "--freeze_bb",
            type=convert_str_2_bool,
            nargs="?",
            const=True,
            default=False,
            help="whether to freeze resnet",
        )

        parser.add_argument(
            "--num_points",
            type=int,
            default=5000,
            help="number of points input to point architecture",
        )

        parser.add_argument(
            "--neg_samples", type=int, default=2000, help="number of negative samples"
        )

        parser.add_argument(
            "--include_depth",
            type=convert_str_2_bool,
            nargs="?",
            const=True,
            default=False,
            help="whether to include depth map or not",
        )

        parser.add_argument(
            "--include_normal",
            type=convert_str_2_bool,
            nargs="?",
            const=True,
            default=False,
            help="whether to include normal map or not",
        )

        parser.add_argument(
            "--include_color",
            type=convert_str_2_bool,
            nargs="?",
            const=True,
            default=False,
            help="whether to include normal map or not",
        )

        parser.add_argument(
            "--flip_normal",
            type=convert_str_2_bool,
            nargs="?",
            const=False,
            default=False,
            help="whether to augment normal maps by flipping the axis",
        )

        parser.add_argument(
            "--normal_to_local",
            type=convert_str_2_bool,
            nargs="?",
            const=False,
            default=False,
            help="whether to augment normal maps by flipping the axis",
        )
        parser.add_argument(
            "--num_classes", type=int, default=13, help="number of negative samples"
        )

        parser.add_argument(
            "--emb_dims",
            type=int,
            default=64,
            metavar="N",
            help="Dimension of embeddings",
        )
        parser.add_argument(
            "--load_imagenet",
            type=convert_str_2_bool,
            nargs="?",
            const=True,
            default=False,
            help="whether to load imagenet pre-trained model",
        )

        parser.add_argument(
            "--train_file_name",
            type=str,
            default="shuffled_train_val_file_list.json",
            help="train + val file names",
        )

        parser.add_argument(
            "--train_label_file_name",
            type=str,
            default="shuffled_train_file_list.json",
            help="file name for labeled dataset",
        )

        parser.add_argument(
            "--category",
            type=str,
            default="",
            help="categories used for training and testing, if all then all categories are selected",
        )

        parser.add_argument(
            "--texture_norm",
            type=str,
            default="gray",
            help="what kind of normalization to use for the dataset",
        )

        parser.add_argument(
            "--voting",
            type=str,
            default="majority",
            help="voting scheme",
        )

        args = parser.parse_args()
        self.args = args

        self.args.model_name = "{}_r_{}_b_{}_lr_{}_g_{}_r_{}_s_{}_upl_{}_f_{" \
                               "}_L_{}_e_{}_n_{}_f_{}_G_{}_A_{}_ID_{}_IN_{" \
                               "}_FN_{}_AG_{}_LI_{}_NL_{}_NP_{}_SP_{}_CL_{" \
                               "}_IC_{}_V_{}_E_{}".format(
            self.args.exp_name,
            self.args.run_id,
            self.args.batch_size,
            self.args.lr,
            self.args.multi_gpus,
            self.args.resume,
            self.args.sampling,
            self.args.up_layers,
            self.args.few_shots,
            self.args.lamb,
            self.args.emb_dims,
            self.args.up_seg_layers,
            self.args.freeze_bb,
            self.args.gamma,
            self.args.all_imgs,
            self.args.include_depth,
            self.args.include_normal,
            self.args.flip_normal,
            self.args.augment,
            self.args.load_imagenet,
            self.args.normal_to_local,
            self.args.num_points,
            self.args.neg_samples,
            self.args.num_classes,
            self.args.include_color,
            self.args.few_images,
            self.args.epochs
        )
        self._set_random_seed(self.args.run_id)

    def _get_args(self):
        return self.args

    def _set_random_seed(self, seed=0):
        """
        Set initial random seeds to ensure reproducibility.
        The seed is selected based on the id of the random
        experiment.
        :param seed: random seed id
        :return:
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
