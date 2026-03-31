import os
from pathlib import Path

import cv2
import numpy as np
import torch

from layers.functions.prior_box import PriorBox
from utils.box_utils import decode
from utils.nms.py_cpu_nms import py_cpu_nms, py_cpu_soft_nms


def parse_retinaface_label_file(label_path):
    """解析 RetinaFace 风格的 label.txt，返回图片路径与对应真值框。"""
    samples = []
    current_boxes = []
    current_image = None

    with open(label_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("#"):
                if current_image is not None:
                    samples.append(
                        {
                            "image_path": os.path.join(label_path.replace("label.txt", "images"), current_image),
                            "boxes": np.array(current_boxes, dtype=np.float32)
                            if current_boxes
                            else np.zeros((0, 4), dtype=np.float32),
                            "image_name": current_image,
                        }
                    )
                current_image = line[2:]
                current_boxes = []
                continue

            values = [float(x) for x in line.split(" ") if x != ""]
            x1 = values[0]
            y1 = values[1]
            x2 = values[0] + values[2]
            y2 = values[1] + values[3]
            current_boxes.append([x1, y1, x2, y2])

    if current_image is not None:
        samples.append(
            {
                "image_path": os.path.join(label_path.replace("label.txt", "images"), current_image),
                "boxes": np.array(current_boxes, dtype=np.float32)
                if current_boxes
                else np.zeros((0, 4), dtype=np.float32),
                "image_name": current_image,
            }
        )

    return samples


def preprocess_image(image):
    """按 RetinaFace 推理方式预处理输入图像。"""
    image = np.float32(image)
    image -= (104, 117, 123)
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).unsqueeze(0)


def run_detector(net, cfg, image_bgr, device, confidence_threshold=0.02, nms_threshold=0.4, top_k=5000, keep_top_k=750):
    """对单张图片执行检测，并返回后处理后的预测框与分数。"""
    im_height, im_width, _ = image_bgr.shape
    scale = torch.tensor([im_width, im_height, im_width, im_height], device=device, dtype=torch.float32)
    image_tensor = preprocess_image(image_bgr).to(device)

    with torch.no_grad():
        loc, conf, _ = net(image_tensor)
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, cfg["variance"])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    if boxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    if cfg.get("use_soft_nms", False):
        keep = py_cpu_soft_nms(dets, nms_threshold)
    else:
        keep = py_cpu_nms(dets, nms_threshold)

    dets = dets[keep, :]
    dets = dets[:keep_top_k, :]
    return dets[:, :4], dets[:, 4]


def compute_iou_matrix(boxes1, boxes2):
    """计算两组框之间的 IoU 矩阵。"""
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def match_detections(pred_boxes, pred_scores, gt_boxes, iou_threshold):
    """对单张图像执行预测框与真值框匹配。"""
    if pred_boxes.shape[0] == 0:
        # 返回顺序固定为：records, num_gt, tp, fp, fn
        return [], int(gt_boxes.shape[0]), 0, 0, int(gt_boxes.shape[0])

    order = pred_scores.argsort()[::-1]
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]

    matched_gt = np.zeros(gt_boxes.shape[0], dtype=bool)
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    records = []
    tp = 0
    fp = 0
    for pred_idx in range(pred_boxes.shape[0]):
        score = float(pred_scores[pred_idx])
        if gt_boxes.shape[0] == 0:
            records.append((score, 0))
            fp += 1
            continue

        gt_idx = int(np.argmax(iou_matrix[pred_idx]))
        best_iou = iou_matrix[pred_idx, gt_idx]
        if best_iou >= iou_threshold and not matched_gt[gt_idx]:
            matched_gt[gt_idx] = True
            records.append((score, 1))
            tp += 1
        else:
            records.append((score, 0))
            fp += 1

    fn = int(gt_boxes.shape[0] - tp)
    return records, int(gt_boxes.shape[0]), int(tp), int(fp), fn


def compute_ap(records, num_gt):
    """根据所有预测记录计算 AP。"""
    if num_gt == 0 or len(records) == 0:
        return 0.0

    records = sorted(records, key=lambda x: x[0], reverse=True)
    tp = np.array([item[1] for item in records], dtype=np.float32)
    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / max(float(num_gt), 1e-12)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)


def compute_metrics(records, num_gt, total_tp, total_fp, total_fn):
    """计算 AP、Precision、Recall、F1-score。"""
    ap = compute_ap(records, num_gt)
    precision = total_tp / max(total_tp + total_fp, 1e-12)
    recall = total_tp / max(total_tp + total_fn, 1e-12)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "ap": float(ap),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "num_gt": int(num_gt),
        "num_pred": int(total_tp + total_fp),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
    }


def save_prediction_txt(prediction_dir, image_name, pred_boxes, pred_scores):
    """将逐图像预测结果保存为 txt，便于后续可视化和排查。"""
    prediction_dir = Path(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    save_path = prediction_dir / (Path(image_name).stem + ".txt")

    with open(save_path, "w", encoding="utf-8") as f:
        for box, score in zip(pred_boxes, pred_scores):
            f.write(
                "{:.2f} {:.2f} {:.2f} {:.2f} {:.6f}\n".format(
                    box[0], box[1], box[2], box[3], score
                )
            )


def save_report(report_path, metrics, eval_cfg):
    """将最终验证结果保存为文本报告。"""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("自建数据集验证结果\n")
        f.write("=" * 40 + "\n")
        f.write(f"网络配置: {eval_cfg['network']}\n")
        f.write(f"模型权重: {eval_cfg['trained_model']}\n")
        f.write(f"标注文件: {eval_cfg['label_file']}\n")
        f.write(f"IoU 阈值: {eval_cfg['iou_threshold']}\n")
        f.write(f"置信度阈值: {eval_cfg['confidence_threshold']}\n")
        f.write(f"NMS 阈值: {eval_cfg['nms_threshold']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"AP@{eval_cfg['iou_threshold']:.2f}: {metrics['ap']:.6f}\n")
        f.write(f"Precision: {metrics['precision']:.6f}\n")
        f.write(f"Recall: {metrics['recall']:.6f}\n")
        f.write(f"F1-score: {metrics['f1']:.6f}\n")
        f.write(f"GT 数量: {metrics['num_gt']}\n")
        f.write(f"预测框数量: {metrics['num_pred']}\n")
        f.write(f"TP: {metrics['tp']}\n")
        f.write(f"FP: {metrics['fp']}\n")
        f.write(f"FN: {metrics['fn']}\n")
