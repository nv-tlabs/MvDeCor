# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import json
import random

import h5py
import numpy as np
from torch.utils.data import IterableDataset

from src.loss import compute_overlap, compute_overlap_abs

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


def load_images(path, shape_id, image_ids, load_face_labels=False):
    point_images = []
    triangles = []
    rendered_images = []
    normals = []
    depths = []
    path = path + shape_id + "/"
    masks = []

    try:
        for img_id in image_ids:
            img_id = str(img_id)
            img = (
                    np.load(path + "rendered_image_{}.npy".format(img_id))[:, :, 0:3]
                    / 255.0
            )
            rendered_images.append(img.astype(np.float32))
            tri = np.load(path + "tri_image_{}.npy".format(img_id)).astype(np.int32)
            depth_img = np.load(path + "depth_image_{}.npy".format(img_id)) / 255.0
            depth_img = np.stack([depth_img] * 3, 2).astype(np.float32)
            normal_img = np.load(path + "normal_image_{}.npy".format(img_id))[
                         :, :, 0:3
                         ].astype(np.float32)

            point_img = np.load(path + "p_image_{}.npy".format(img_id))[
                        :, :, 0:3
                        ].astype(np.float32)
            x, y = np.where(tri > -1)
            masks.append([x, y])
            depths.append(depth_img)
            normals.append(normal_img)
            triangles.append(tri)
            point_images.append(point_img)
    except:
        print("Error in loading images {} for shape id {}".format(image_ids, shape_id))

    if load_face_labels:
        face_labels = np.load(path + "faceid.npy").astype(np.int32)
        data = {
            "img": rendered_images,
            "normal": normals,
            "tri": triangles,
            "depth": depths,
            "points": point_images,
            "id": [shape_id] * 2,
            "labels": face_labels,
            "mask": masks,
        }
    else:
        data = {
            "img": rendered_images,
            "normal": normals,
            "tri": triangles,
            "depth": depths,
            "points": point_images,
            "id": [shape_id] * 2,
            "mask": masks,
        }
    return data


def normalize_image(images, img_type="gray"):
    """
    Normalize image with mean and std
    :param images: N x H x W x 3 image batch
    """
    mean = MEANS[img_type]
    std = STDS[img_type]
    images = images - mean.reshape((1, 1, 1, 3))
    images = images / std.reshape((1, 1, 1, 3))
    return images


def data_load(path):
    """
    Loads a shape data from the disc
    :param path: path to the data root
    :return: loaded raw data per shape
    """
    # load all images
    triangles = []
    rendered_images_ = []
    normals = []
    depths = []
    p_images = []
    for i in range(86):
        img = np.load(path + "rendered_image_{}.npy".format(i))[:, :, 0:3] / 255.0
        tri = np.load(path + "tri_image_{}.npy".format(i))
        depth_img = np.load(path + "depth_image_{}.npy".format(i)) / 255.0
        depth_img = np.stack([depth_img] * 3, 2)
        normal_img = np.load(path + "normal_image_{}.npy".format(i))[:, :, 0:3]
        p_image = np.load(path + "p_image_{}.npy".format(i))[:, :, 0:3]

        rendered_images_.append(img)
        depths.append(depth_img)
        normals.append(normal_img)
        triangles.append(tri)
        p_images.append(p_image)
    triangles = np.stack(triangles, 0)
    rendered_images = normalize_image(np.stack(rendered_images_, 0), "gray")
    depths = normalize_image(np.stack(depths, 0), "depth")
    normals = normalize_image(np.stack(normals, 0), "normal")
    return rendered_images_, rendered_images, depths, normals, triangles, p_images


def gather_images(data):
    new_data = {}
    for k, v in data[0].items():
        new_data[k] = []

    for d in data:
        for k, v in d.items():
            new_data[k] += v
    return new_data


def check_validity(pair):
    return compute_overlap(pair["points"][0], pair["points"][1])


def check_validity_abs(pair):
    return compute_overlap_abs(pair["points"][0], pair["points"][1])


def assemble_segments(
        pairs, retain_original=False, load_labels=False, include_list=None
):
    triangles = []
    rendered_images = []
    normals = []
    depths = []
    point_images = []
    ids = []
    if load_labels:
        labels = []
    masks = []
    for pair in pairs:
        triangles.append(pair["tri"][0])
        rendered_images.append(pair["img"][0])
        normals.append(pair["normal"][0])
        depths.append(pair["depth"][0])
        point_images.append(pair["points"][0])
        masks.append(pair["mask"][0])
        ids += pair["id"]
        if load_labels:
            labels += [pair["labels"]]

    triangles = np.stack(triangles, 0)
    original_images = rendered_images
    rendered_images = normalize_image(np.stack(rendered_images, 0), "gray")
    depths = normalize_image(np.stack(depths, 0), "depth")
    normals = normalize_image(np.stack(normals, 0), "normal")
    data = {"tri": triangles, "points": point_images}
    # inputs = np.concatenate([rendered_images, depths, normals], 3).astype(np.float32)
    inputs = [rendered_images]
    if "depth" in include_list:
        inputs.append(depths)
    if "normal" in include_list:
        inputs.append(normals)

    inputs = np.concatenate(inputs, 3).astype(np.float32)
    inputs = inputs.transpose((0, 3, 1, 2))
    data["inputs"] = inputs
    data["ids"] = ids
    data["mask"] = masks

    if retain_original:
        data["orig"] = original_images
    if load_labels:
        data["labels"] = labels
    return data


def assemble(pairs, retain_original=False, load_labels=False, include_list=None):
    triangles = []
    rendered_images = []
    normals = []
    depths = []
    point_images = []
    ids = []
    if load_labels:
        labels = []

    for pair in pairs:
        triangles.append(pair["tri"][0])
        triangles.append(pair["tri"][1])
        rendered_images.append(pair["img"][0])
        rendered_images.append(pair["img"][1])
        normals.append(pair["normal"][0])
        normals.append(pair["normal"][1])
        depths.append(pair["depth"][0])
        depths.append(pair["depth"][1])
        point_images.append(pair["points"][0])
        point_images.append(pair["points"][1])
        ids += pair["id"]
        if load_labels:
            labels += [pair["labels"], pair["labels"]]

    triangles = np.stack(triangles, 0)
    original_images = rendered_images
    rendered_images = normalize_image(np.stack(rendered_images, 0), "gray")
    depths = normalize_image(np.stack(depths, 0), "depth")
    normals = normalize_image(np.stack(normals, 0), "normal")
    data = {"tri": triangles, "points": point_images}
    inputs = [rendered_images]

    if "depth" in include_list:
        inputs.append(depths)
    if "normal" in include_list:
        inputs.append(normals)
    inputs = np.concatenate(inputs, 3).astype(np.float32)
    inputs = inputs.transpose((0, 3, 1, 2))
    data["inputs"] = inputs
    data["ids"] = ids
    if retain_original:
        data["orig"] = original_images
    if load_labels:
        data["labels"] = labels
    return data


def find_set(path, cat, set_len, run_id):
    """
    Finds a set of shapes that together covers the entire label set.
    In some categories, this is not possible e.g. Chair, Table etc,
    then simply select the set that covers the label set as much as possible.
    :param path: root path
    :param cat: category of the partnet dataset
    :param set_len: number of shapes needed for the few-shot experiment
    :param run_id: few-shot experiments are done on random subset. run_id is
    an integer corresponding to the id of the experiment.
    """
    path_points = path.format(cat)
    Labels = []
    print(path, cat, set_len, run_id)
    for i in range(100):
        try:
            with h5py.File(path_points + "train-0{}.h5".format(i), "r") as hf:
                label_seg = np.array(hf.get("label_seg"))
            Labels.append(label_seg)
        except:
            break
    Labels = np.concatenate(Labels)
    valid_ids = []
    len_of_set = np.unique(Labels).shape[0]

    count = 0
    iterations = 0
    random_sets = []
    random_sets_len = []
    while True:
        indices = np.random.choice(Labels.shape[0], set_len)
        labels = Labels[indices]
        random_len = np.unique(labels).shape[0]
        random_sets.append(indices)
        random_sets_len.append(random_len)

        if random_len == len_of_set:
            count += 1
            valid_ids.append(indices.tolist())
        if count == 5:
            break

        iterations += 1
        if iterations > 10000:
            break
    try:
        return valid_ids[run_id]
    except:
        for i in np.argsort(random_sets_len)[-5:]:
            valid_ids.append(random_sets[i])
        return valid_ids[run_id]


class SegmentShapenetDataset:
    def __init__(
            self,
            path,
            load_labels=False,
            max_num_images=86,
            include_list=["depth", "normal"],
            if_flip_normals=False,
            convert_normal_to_local=False,
            if_augment=False,
            k_shots=0,
            categories=None,
            seg_path=None,
            train_files="valid_shuffled_train_file_list.json",
            test_files="valid_shuffled_test_file_list.json",
            pre_load=False,
            k_images=-1,
            run_id=0,
            means=None,
            stds=None
    ):
        """
        Dataset to load segments. For each shape, we randomly select a view and its corresponding
        segmentation mask
        :param path: Path to the root where dataset is stored
        :param load_labels: whether to load labels or not
        :param max_num_images: maximum number of images already rendered per shape stored on the disc
        :param include_list: data other than grayscale to be input to the network
        :param k_shots: number of few-shot shapes
        :param categories: categories used to train
        :param seg_path: path to the segmentation dataset
        :param train_files: file path consist of training data ids
        :param test_files: file path consist of testing data ids
        :param pre_load: whether to preload the data
        :param k_images: number of images per shape used to train
        :param run_id: id of the random experiment index
        :param means: means of the data computed over the entire training set
        :param stds: std of the data computed over the entire training set
        """
        self.path = path
        self.load_labels = load_labels
        self.include_list = include_list
        self.flip_normals = if_flip_normals
        self.convert_normal_to_local = convert_normal_to_local
        self.if_augment = if_augment
        self.pre_load = pre_load
        self.k_images = k_images
        self.seg_path = seg_path
        if means:
            global MEANS, STDS
            MEANS = means
            STDS = stds

        self.max_num_images = max_num_images

        # load the train, val and test split files
        path = "{}/extra_files/train_test_split/".format(path)

        with open(path + train_files, "r") as file:
            train_ids = json.load(file)
        with open(path + test_files, "r") as file:
            test_ids = json.load(file)

        train_ids_categories = {}
        test_ids_categories = {}

        for c in categories:
            train_ids_categories[c] = []
            test_ids_categories[c] = []

        for i in train_ids:
            d = i.split("/")
            if len(d) == 2:
                c, id = d
            else:
                c, id = d[1], d[2]
            if c in categories:
                train_ids_categories[c].append(c + "/" + id)

        for i in test_ids:
            d = i.split("/")
            if len(d) == 2:
                c, id = d
            else:
                c, id = d[1], d[2]
            if c in categories:
                test_ids_categories[c].append(c + "/" + id)

        train_ids_categories_images = {}
        # randomly select k examples for each category.
        for c in train_ids_categories.keys():
            if k_shots > -1:
                all_run_ids = find_set(
                    self.seg_path + "/sem_seg_h5/{}-3/", categories[0], k_shots, run_id)
                valid_ids = []
                for k in all_run_ids:
                    valid_ids.append(train_ids_categories[c][k])
                train_ids_categories[c] = valid_ids

            # number of k images, for sparse view experiments
            train_ids_categories_images[c] = {}
            for shape_id in train_ids_categories[c]:
                if self.k_images > -1:
                    train_ids_categories_images[c][shape_id] = np.random.choice(
                        max_num_images, self.k_images)
                else:
                    train_ids_categories_images[c][shape_id] = np.arange(
                        self.max_num_images)

        self.train_ids_categories_images = train_ids_categories_images
        self.train_ids_categories = train_ids_categories
        self.test_ids_categories = test_ids_categories
        self.categories = categories
        print(train_ids_categories, flush=True)
        if self.pre_load:
            self.pre_load_train()

    def pre_load_train(self):
        print("Inside pre loading training")
        all_data = {}
        uniques = np.zeros((100))

        for c_id in self.categories:
            all_data[c_id] = {}
            print (len(all_data[c_id]))
            for shape_id in self.train_ids_categories[c_id]:
                pairs = []
                for i in self.train_ids_categories_images[c_id][shape_id]:
                    image_ids = [i]
                    pair = load_images(
                        self.path,
                        shape_id,
                        image_ids,
                        load_face_labels=self.load_labels,
                    )
                    pairs.append(pair)
                    uniques[np.unique(pair["labels"])] = 1
                all_data[c_id][shape_id] = pairs
        self.all_data = all_data
        print("Unique ids in: ", np.where(uniques), flush=True)
        print("Total uniques in: ", uniques.sum(), flush=True)

    def load_train(self, batch_size):
        while True:
            category_ids = np.random.choice(len(self.categories), batch_size)
            category_ids = [self.categories[c] for c in category_ids]

            pairs = []
            for c_id in category_ids:
                shape_index = np.random.choice(len(self.train_ids_categories[
                                                       c_id]), 1)[0]
                shape_id = self.train_ids_categories[c_id][shape_index]
                image_ids = np.random.choice(self.k_images if self.k_images
                                                              > -1 else
                                             self.max_num_images, 1,
                                             replace=False)
                if not self.pre_load:
                    pair = load_images(
                        self.path,
                        shape_id,
                        image_ids,
                        load_face_labels=self.load_labels,
                    )
                else:
                    pair = self.all_data[c_id][shape_id][image_ids[0]]
                pairs.append(pair)

            # Assemble the data into batches
            data = assemble_segments(
                pairs, load_labels=self.load_labels, include_list=self.include_list
            )
            yield data

    def load_val(self, batch_size):
        while True:
            category_ids = np.random.choice(len(self.categories), batch_size)
            category_ids = [self.categories[c] for c in category_ids]

            pairs = []
            for c_id in category_ids:
                shape_index = np.random.choice(len(self.test_ids_categories[
                                                       c_id]), 1)[0]
                shape_id = self.test_ids_categories[c_id][shape_index]
                image_ids = np.random.choice(self.max_num_images, 1, replace=False)

                pair = load_images(
                    self.path,
                    shape_id,
                    image_ids,
                    load_face_labels=self.load_labels,
                )
                pairs.append(pair)

            # Assemble the data into batches
            data = assemble_segments(
                pairs, load_labels=self.load_labels, include_list=self.include_list
            )
            yield data

    def load_eval(self):
        for c in self.categories:
            for s_id in self.test_ids_categories[c]:
                pairs = []
                for j in range(0, self.max_num_images, 1):
                    image_ids = [j]
                    pair = load_images(
                        self.path,
                        s_id,
                        image_ids,
                        load_face_labels=self.load_labels,
                    )
                    pairs += [pair]
                    # Assemble the data into batches

                data = assemble_segments(
                    pairs,
                    include_list=self.include_list,
                    retain_original=True,
                    load_labels=self.load_labels,
                )
                yield data


class Dataset:
    def __init__(
            self,
            path,
            train_ids,
            val_ids,
            max_num_images=86,
            include_list=["depth", "normal"],
            means=None,
            stds=None,
            pre_load=False,
            pre_load_size=100
    ):
        """
        Dataset class used for pre-training MvDecor. It loads a pair of images
        from a shape, that has matching above a certain threshold.
        :param path: Path to the root where dataset is stored
        :param train_ids: training shape ids
        :param val_ids: validation shape ids
        :param max_num_images: maximum number of images already rendered per shape stored on the disc
        :param include_list: data other than grayscale to be input to the network
        :param means: means of the data computed over the entire training set
        :param stds: std of the data computed over the entire training set
        :param pre_load: whether to preload some data
        :param pre_load_size: size of the preload
        """
        self.train_ids = np.array(train_ids)
        self.val_ids = np.array(val_ids)
        self.path = path
        self.max_num_images = max_num_images
        self.include_list = include_list
        self.camera_poses = None
        self.pre_load = pre_load
        if means:
            global MEANS, STDS
            MEANS = means
            STDS = stds

        if self.pre_load:
            self.pre_load_train(pre_load_size)

    def pre_load_train(self, len=4):
        all_data = {}
        import copy
        train_ids = copy.deepcopy(self.train_ids)
        random.shuffle(train_ids)

        for index, shape_id in enumerate(train_ids[0:len]):
            print (index, shape_id)
            pairs = []
            for i in range(0, self.max_num_images):
                image_ids = [i]
                pair = load_images(
                    self.path,
                    shape_id,
                    image_ids,
                )
                pairs.append(pair)

            all_data[shape_id] = pairs
        self.all_data = all_data

    def load_train(self, batch_size, cudafy=False):
        while True:
            pairs = []
            while True:
                if self.pre_load:
                    train_ids = list(self.all_data.keys())
                else:
                    train_ids = self.train_ids
                shape_ids = np.random.choice(len(train_ids), 1)
                s_id = train_ids[shape_ids[0]]

                while True:
                    image_ids = np.random.choice(self.max_num_images, 2, replace=False)
                    if self.pre_load:
                        if s_id in self.all_data.keys():
                            pair = gather_images([self.all_data[s_id][
                                                      image_ids[0]], \
                                                  self.all_data[s_id][image_ids[1]]])
                    else:
                        pair = load_images(
                            self.path,
                            s_id,
                            image_ids,
                        )

                    if len(pair["points"]) < 2:
                        break
                    if not check_validity(pair):
                        continue
                    else:
                        pairs.append(pair)
                        break

                if len(pairs) == batch_size // 2:
                    break

            # Assemble the data into batches
            data = assemble(pairs, include_list=self.include_list)
            yield data

    def load_val(self, batch_size, cudafy=False):
        while True:
            pairs = []
            while True:
                shape_ids = np.random.choice(len(self.val_ids), 1)
                s_id = self.val_ids[shape_ids[0]]

                while True:
                    image_ids = np.random.choice(self.max_num_images, 2, replace=False)
                    pair = load_images(
                        self.path,
                        s_id,
                        image_ids,
                    )
                    if len(pair["points"]) < 2:
                        break
                    if not check_validity(pair):
                        continue
                    else:
                        pairs.append(pair)
                        break

                if len(pairs) == batch_size // 2:
                    break

            # Assemble the data into batches
            data = assemble(pairs, include_list=self.include_list)
            yield data

    def load_eval(self, batch_size, cudafy=False):
        for s in self.val_ids:
            for i in range(0, self.max_num_images, 10):
                for j in range(0, self.max_num_images, 10):
                    if i == j:
                        continue
                    image_ids = [i, j]

                    pair = load_images(self.path, s, image_ids)
                    validity = check_validity_abs(pair)
                    if not (validity >= 0.15):
                        continue
                    pairs = [pair]

                    # Assemble the data into batches
                    data = assemble(pairs, retain_original=True)
                    data["overlap"] = validity
                    yield data


class GeneratorIter(IterableDataset):
    def __init__(self, gen, size):
        self.gen = gen
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.gen)


def initialize_workers(id):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="big") + id)
