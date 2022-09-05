# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import os

import numpy as np
import torch
import trimesh
from torch.utils.data import DataLoader

from src.dataset import GeneratorIter, initialize_workers
from src.loss import SegmentationLoss
from src.partnet_eval_utils import calculate_iou, \
    calculate_shape_iou, calculate_part_iou
from src.render_utils import find_match


def test_partnet(net, opt, dataset, model_name, voting="entropy"):
    Loss = SegmentationLoss()
    NUM_SEG = {'Bed': 15,
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

    shape_ious = {opt.category: []}
    dataset.all_data = None

    val_loader = dataset.load_eval()

    def collate_fn(x):
        return x

    val_loader = GeneratorIter(val_loader, int(1e10))
    val_loader = iter(
        DataLoader(
            val_loader,
            batch_size=1,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=1,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=initialize_workers,
        )
    )

    net.eval()

    os.makedirs("../logs_selfsup-proj/results/{}/".format(model_name),
                exist_ok=True)
    save_path = "../logs_selfsup-proj/results/{}/".format(model_name)

    IOUS = {}
    for j, data in enumerate(val_loader):
        data = data[0]

        with torch.no_grad():
            outputs = []
            for i in range(len(data["inputs"])):
                output = net(torch.from_numpy(data["inputs"][i: i + 1]).cuda())
                outputs.append(output)
            output = torch.cat(outputs, 0)
            outputs = [output]

        predicted_tri_labels = Loss.accuracy_3d(outputs, data,
                                                exclude_zero=True)
        print(predicted_tri_labels.shape, data["labels"][0].shape, data["ids"][0], flush=True)
        np.save("{}predtri_{}.npy".format(save_path, data["ids"][0].split("/")[1]),
                predicted_tri_labels.astype(np.int16))

        mesh_path = "{}{}".format(opt.data_dir, data["ids"][0])
        mesh = trimesh.load(mesh_path + "/" + "mesh.obj", force="mesh")

        face_labels = np.load(mesh_path + "/" + "faceid.npy")

        valid_indices = predicted_tri_labels > -1
        invalid_indices = predicted_tri_labels == -1
        if invalid_indices.sum() > 0:
            _, indices = find_match(mesh.triangles_center[valid_indices],
                                    mesh.triangles_center[invalid_indices])
            predicted_tri_labels[invalid_indices] = predicted_tri_labels[valid_indices][indices]
        print(np.sum(predicted_tri_labels == 0), predicted_tri_labels.shape,
              np.sum(face_labels == 0))
        # Adjust the zero label
        predicted_tri_labels[face_labels == 0] = 0

        mesh_sampled_points, point_index = trimesh.sample.sample_surface(
            mesh, 30000)

        per_point_labels = predicted_tri_labels[point_index]
        gt_labels_at_sampled_points = face_labels[point_index]

        IOUS[data["ids"][0]] = calculate_iou(gt_labels_at_sampled_points,
                                             per_point_labels,
                                             NUM_SEG[opt.category])
        print(calculate_part_iou(IOUS, NUM_SEG[opt.category]), flush=True)

        torch.cuda.empty_cache()
        del outputs
    shape_ious = calculate_shape_iou(IOUS)
    part_ious = calculate_part_iou(IOUS, NUM_SEG[opt.category])

    test_metrics = {"shape_iou": shape_ious, "part_iou": part_ious}
    print(test_metrics, flush=True)

    with open(save_path + "result_{}.org".format(voting), "w") as file:
        json.dump(test_metrics, file)
    return test_metrics
