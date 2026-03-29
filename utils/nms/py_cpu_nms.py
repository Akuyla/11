# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def py_cpu_soft_nms(dets, thresh, sigma=0.5, score_thresh=0.001, method='linear'):
    """Pure Python Soft-NMS baseline."""
    if dets.shape[0] == 0:
        return []

    dets = dets.copy()
    scores = dets[:, 4]
    indexes = np.arange(dets.shape[0])
    keep = []

    while dets.shape[0] > 0:
        max_idx = np.argmax(scores)
        dets[[0, max_idx]] = dets[[max_idx, 0]]
        scores[[0, max_idx]] = scores[[max_idx, 0]]
        indexes[[0, max_idx]] = indexes[[max_idx, 0]]

        keep.append(indexes[0])
        if dets.shape[0] == 1:
            break

        x1 = np.maximum(dets[0, 0], dets[1:, 0])
        y1 = np.maximum(dets[0, 1], dets[1:, 1])
        x2 = np.minimum(dets[0, 2], dets[1:, 2])
        y2 = np.minimum(dets[0, 3], dets[1:, 3])

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        inter = w * h

        area0 = (dets[0, 2] - dets[0, 0] + 1) * (dets[0, 3] - dets[0, 1] + 1)
        area1 = (dets[1:, 2] - dets[1:, 0] + 1) * (dets[1:, 3] - dets[1:, 1] + 1)
        iou = inter / (area0 + area1 - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > thresh] -= iou[iou > thresh]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:
            weight = np.ones_like(iou)
            weight[iou > thresh] = 0.0

        scores[1:] = scores[1:] * weight
        keep_mask = scores[1:] > score_thresh

        dets = dets[1:][keep_mask]
        scores = scores[1:][keep_mask]
        indexes = indexes[1:][keep_mask]

    return keep
