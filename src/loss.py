# SPDX-FileCopyrightText: Copyright (c) <2022> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import torch

from src.render_utils import find_match


def find_correspondence_bw_images(triangle_ids1, triangle_ids2, return_outside=False):
    if len(triangle_ids1.shape) == 2:
        triangle_ids1 = np.expand_dims(triangle_ids1, 2)
        triangle_ids2 = np.expand_dims(triangle_ids2, 2)

    x_1, y_1 = np.where(triangle_ids1[:, :, 0] > -1)
    triangle_index_1 = triangle_ids1[x_1, y_1]
    x_2, y_2 = np.where(triangle_ids2[:, :, 0] > -1)
    triangle_index_2 = triangle_ids2[x_2, y_2]
    d, indices = find_match(triangle_index_1, triangle_index_2)
    matched_indices_2 = np.where(d < 5e-3)[0]

    matched_indices_1 = indices[matched_indices_2]

    matched_x_2 = x_2[matched_indices_2]
    matched_y_2 = y_2[matched_indices_2]

    matched_x_1 = x_1[matched_indices_1]
    matched_y_1 = y_1[matched_indices_1]

    if return_outside:
        outside_indices_2 = np.where(d > 8e-3)[0]
        return (
            matched_x_1,
            matched_y_1,
            matched_x_2,
            matched_y_2,
            triangle_ids1[matched_x_1, matched_y_1][:, 0],
            triangle_ids2[matched_x_2, matched_y_2][:, 0],
            x_2[outside_indices_2],
            y_2[outside_indices_2],
        )

    return (
        matched_x_1,
        matched_y_1,
        matched_x_2,
        matched_y_2,
        triangle_ids1[matched_x_1, matched_y_1][:, 0],
        triangle_ids2[matched_x_2, matched_y_2][:, 0],
    )


def compute_loss_batch(outputs, triangles, strategy=0, neg_samples=4000):
    b = outputs.shape[0]
    losses = []
    for i in range(0, b, 2):
        l = loss_infonce(
            outputs[i: i + 2],
            triangles[i: i + 2],
            max_matches=neg_samples,
            strategy=strategy,
        )
        torch.cuda.empty_cache()
        losses.append(l)
    losses = torch.stack(losses).mean()
    return losses


def loss_infonce(output, tri, T=0.07, max_matches=None, strategy=0):
    tri1 = tri[0]
    tri2 = tri[1]

    out1 = output[0].permute((1, 2, 0))
    out2 = output[1].permute((1, 2, 0))

    loss1 = one_side_infonce_loss(
        tri1, tri2, out1, out2, max_matches=max_matches, strategy=strategy
    )
    loss2 = one_side_infonce_loss(
        tri2, tri1, out2, out1, max_matches=max_matches, strategy=strategy
    )
    loss = (loss1 + loss2) / 2.0
    return loss


def shuffle_columns_independent(mat):
    h, w = mat.shape
    ind = torch.randint(0, w, (h, w), device=mat.device)
    out = torch.zeros((h, w), device=mat.device).scatter_(1, ind, mat)
    return out


def one_side_infonce_loss(tri1, tri2, out1, out2, T=0.07, max_matches=None, strategy=0):
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    x_1, y_1, x_2, y_2, tid1, tid2 = find_correspondence_bw_images(tri1, tri2)

    if max_matches:
        random_indices = np.random.choice(x_1.shape[0], max_matches)
        x_1 = x_1[random_indices]
        x_2 = x_2[random_indices]
        y_1 = y_1[random_indices]
        y_2 = y_2[random_indices]

    matched_out1 = out1[x_1, y_1]
    matched_out2 = out2[x_2, y_2]
    matched_out1 = torch.nn.functional.normalize(matched_out1, p=2, dim=-1)
    matched_out2 = torch.nn.functional.normalize(matched_out2, p=2, dim=-1)

    logits = torch.mm(matched_out1, matched_out2.transpose(1, 0))  # npos by npos
    labels = torch.arange(x_1.shape[0]).cuda().long()
    out = torch.div(logits, T)

    loss = criterion(out, labels)
    return loss


def compute_overlap(tri1, tri2):
    if len(tri1.shape) == 2:
        tri1 = np.expand_dims(tri1, 2)
        tri2 = np.expand_dims(tri2, 2)

    x1, y1, x2, y2, tid1, tid2 = find_correspondence_bw_images(tri1, tri2)
    threshold = x1.shape[0] / np.min(
        [np.sum(tri1[:, :, 0] > -1), np.sum(tri2[:, :, 0] > -1)]
    )
    return threshold > 0.15


def compute_overlap_matching(tri1, tri2):
    if len(tri1.shape) == 2:
        tri1 = np.expand_dims(tri1, 2)
        tri2 = np.expand_dims(tri2, 2)

    x1, y1, x2, y2, tid1, tid2 = find_correspondence_bw_images(tri1, tri2)
    threshold = x1.shape[0] / np.min(
        [np.sum(tri1[:, :, 0] > -1), np.sum(tri2[:, :, 0] > -1)]
    )
    return threshold > 0.15, x1, y1, x2, y2


def compute_overlap_abs(tri1, tri2):
    if len(tri1.shape) == 2:
        tri1 = np.expand_dims(tri1, 2)
        tri2 = np.expand_dims(tri2, 2)

    x1, y1, x2, y2, tid1, tid2 = find_correspondence_bw_images(tri1, tri2)
    threshold = x1.shape[0] / np.min(
        [np.sum(tri1[:, :, 0] > -1), np.sum(tri2[:, :, 0] > -1)]
    )
    return threshold


def entropy(logits):
    p = torch.nn.Softmax(dim=1)(logits) + 1e-6
    entropy = - torch.mean(torch.sum(p * torch.log(p), 1))
    return entropy


class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, outputs, triangles, sampling):
        l = compute_loss_batch(outputs, triangles, sampling)
        return l


class SegmentationLoss:
    def __init__(self, weights=None):
        self.criterion = torch.nn.CrossEntropyLoss(weight=weights)

    def forward(self, outputs, data):
        outputs = outputs.permute((0, 2, 3, 1))
        triangles = data["tri"]
        labels = data["labels"]
        batch_size = len(outputs)
        losses = []
        for j in range(batch_size):
            loss = self.compute_segment_loss(outputs[j], triangles[j], labels[j])
            losses.append(loss)
        return torch.stack(losses).mean()

    def compute_segment_loss(self, outputs, triangles, labels):
        x, y = np.where(triangles > -1)

        out = outputs[x, y]
        out_labels = torch.from_numpy(labels[triangles[x, y]]).long().cuda()
        return self.criterion(out, out_labels)

    def accuracy_batch(self, outputs, data, max_labels=None):
        outputs = outputs.permute((0, 2, 3, 1))
        batch_size = len(outputs)
        triangles = data["tri"]
        labels = data["labels"]
        precisions = []
        class_avg = []
        for b in range(batch_size):
            precision = self.accuracy(
                outputs[b], triangles[b], labels[b], max_labels=max_labels
            )
            precisions.append(precision[0])
            class_avg.append(precision[1])
        return precisions, class_avg

    def accuracy(self, outputs, triangles, labels, max_labels=None):
        x, y = np.where(triangles > -1)
        labels = labels[triangles[x, y]]
        out = outputs[x, y]
        predicted_labels = torch.max(out, 1)[1].data.cpu().numpy()
        precision = np.mean(labels == predicted_labels)
        class_avg = self.class_avg(predicted_labels, labels, max_labels=max_labels)
        return precision, class_avg

    def class_avg(self, predicted_labels, labels, max_labels):
        ious = []
        for c in range(max_labels):
            gt_indices = labels == c
            predicted_indices = predicted_labels == c
            i = np.logical_and(gt_indices, predicted_indices).sum()
            u = np.logical_or(gt_indices, predicted_indices).sum()
            iou = i / (u + 1e-5)
            if u > 0:
                ious.append(iou)
        return np.mean(ious)

    def accuracy_3d(self, outputs, data, exclude_zero=False):
        tri_labels = np.ones_like(data["labels"][0]) * -1
        outputs = torch.cat(outputs, 0)
        outputs = outputs.permute((0, 2, 3, 1))
        masks = []
        predicted_labels = []
        predicted_tri_ids = []
        entropies = []
        weighted_predictions = []
        for i in range(len(data["tri"])):
            x, y = np.where(data["tri"][i] > -1)
            masks.append([x, y])
            H = entropy(outputs[i][x, y])

            if exclude_zero:
                # exlclude zero label for partnet experiments because this
                # label corresponds to indetemined points.
                outputs[i][x, y][:, 0] = torch.min(outputs[i][x, y], 1)[0]

                pred_label = torch.max(outputs[i][x, y][:, 1:], 1)[1].data.cpu(

                ).numpy() + 1
            else:
                pred_label = torch.max(outputs[i][x, y], 1)[1].data.cpu().numpy()

            predicted_labels.append(pred_label)

            entropies.append(np.ones_like(pred_label) * (1 - H.item()))

            weighted_predictions.append(outputs[i][x, y] * (1 - H) ** 20)
            predicted_tri_ids.append(data["tri"][i][x, y])

        predicted_labels = np.concatenate(predicted_labels)
        predicted_tri_ids = np.concatenate(predicted_tri_ids)
        entropies = np.concatenate(entropies)
        weighted_predictions = torch.cat(weighted_predictions)

        uniques = np.unique(predicted_tri_ids)

        for u in uniques:
            tri_labels[u] = torch.max(torch.sum(weighted_predictions[
                                                    predicted_tri_ids ==
                                                    u], 0), 0)[1].item()
        return tri_labels
