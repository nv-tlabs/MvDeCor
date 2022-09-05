# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import numpy as np


def calculate_iou(ground, prediction, num_labels):
    """
    Computes IOU of point cloud
    :param ground: numpy array consisting of ground truth labels
    :param prediction: numpy array of predicted labels
    :param num_labels: int, total number of labels
    :return: 
    """

    label_iou, intersection, union = {}, {}, {}
    # Ignore undetermined
    prediction = np.copy(prediction)
    prediction[ground == 0] = 0

    for i in range(1, num_labels):
        # Calculate intersection and union for ground truth and predicted labels
        intersection_i = np.sum((ground == i) & (prediction == i))
        union_i = np.sum((ground == i) | (prediction == i))

        # If label i is present either on the gt or the pred set
        if union_i > 0:
            intersection[i] = float(intersection_i)
            union[i] = float(union_i)
            label_iou[i] = intersection[i] / union[i]

    metrics = {"label_iou": label_iou, "intersection": intersection, "union": union}

    return metrics


def calculate_shape_iou(ious):
    """
    Computes average shape IOU    
    :param ious: dictionary containing for each shape its label iou,
     intersection and union scores with respect to the ground truth.  
    :return: 
      aveg_shape_IOU: float
    """
    shape_iou = {}

    for model_name, metrics in ious.items():
        # Average label iou per shape
        L_s = len(metrics["label_iou"])
        shape_iou[model_name] = np.nan_to_num(np.sum([v for v in metrics["label_iou"].values()]) / float(L_s))

    # Dataset avg shape iou
    avg_shape_iou = np.sum([v for v in shape_iou.values()]) / float(len(ious))

    return avg_shape_iou


def calculate_part_iou(ious, num_labels):
    """
    Calculates part IOU
    :param ious: dictionary containing for each shape its label iou,
     intersection and union scores with respect to the ground truth.  
    :param num_labels: int, total number of labels in the category
    :return: 
      aveg_shape_IOU: float
    """
    intersection = {i: 0.0 for i in range(1, num_labels)}
    union = {i: 0.0 for i in range(1, num_labels)}

    for model_name, metrics in ious.items():
        for label in metrics["intersection"].keys():
            # Accumulate intersection and union for each label across all shapes
            intersection[label] += metrics["intersection"][label]
            union[label] += metrics["union"][label]

    # Calculate part IOU for each label
    part_iou = {}
    for key in range(1, num_labels):
        try:
            part_iou[key] = intersection[key] / union[key]
        except ZeroDivisionError:
            part_iou[key] = 0.0
    # Avg part IOU
    avg_part_iou = np.sum([v for v in part_iou.values()]) / float(num_labels - 1)

    return avg_part_iou
