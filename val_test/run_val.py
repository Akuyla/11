import sys
from pathlib import Path

import cv2
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data import (  # noqa: E402
    cfg_resnest50,
    cfg_resnest50_atss,
    cfg_resnest50_p2,
    cfg_resnest50_p2_se,
    cfg_resnest50_p2_atss_se,
    cfg_resnest50_p2_atss_se_softnms,
)
from models.retinaface import RetinaFace  # noqa: E402
from val_test.custom_val_config import ACTIVE_VAL_CONFIG  # noqa: E402
from val_test.eval_utils import (  # noqa: E402
    compute_metrics,
    match_detections,
    parse_retinaface_label_file,
    run_detector,
    save_prediction_txt,
    save_report,
)


def get_cfg(network_name):
    """根据网络名返回对应训练配置。"""
    if network_name == "resnest50":
        return cfg_resnest50
    if network_name == "resnest50_atss":
        return cfg_resnest50_atss
    if network_name == "resnest50_p2":
        return cfg_resnest50_p2
    if network_name == "resnest50_p2_se":
        return cfg_resnest50_p2_se
    if network_name == "resnest50_p2_atss_se":
        return cfg_resnest50_p2_atss_se
    if network_name == "resnest50_p2_atss_se_softnms":
        return cfg_resnest50_p2_atss_se_softnms
    raise ValueError("Unsupported network: {}".format(network_name))


def check_keys(model, pretrained_state_dict):
    """检查权重与模型参数是否能正确匹配。"""
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print("Missing keys:{}".format(len(missing_keys)))
    print("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    print("Used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """去掉 DataParallel 保存权重时附带的 module. 前缀。"""
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    """加载训练完成后的 pth 权重。"""
    print("Loading pretrained model from {}".format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def main():
    eval_cfg = ACTIVE_VAL_CONFIG
    cfg = get_cfg(eval_cfg["network"])

    device = torch.device("cpu" if eval_cfg["cpu"] else "cuda")

    net = RetinaFace(cfg=cfg, phase="test")
    net = load_model(net, eval_cfg["trained_model"], eval_cfg["cpu"])
    net.eval()
    net = net.to(device)

    samples = parse_retinaface_label_file(eval_cfg["label_file"])
    print("验证样本数: {}".format(len(samples)))

    all_records = []
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for index, sample in enumerate(samples):
        image_bgr = cv2.imread(sample["image_path"])
        if image_bgr is None:
            raise ValueError("图像读取失败: {}".format(sample["image_path"]))

        pred_boxes, pred_scores = run_detector(
            net=net,
            cfg=cfg,
            image_bgr=image_bgr,
            device=device,
            confidence_threshold=eval_cfg["confidence_threshold"],
            nms_threshold=eval_cfg["nms_threshold"],
            top_k=eval_cfg["top_k"],
            keep_top_k=eval_cfg["keep_top_k"],
        )

        if eval_cfg["save_predictions"]:
            save_prediction_txt(
                prediction_dir=eval_cfg["prediction_dir"],
                image_name=sample["image_name"],
                pred_boxes=pred_boxes,
                pred_scores=pred_scores,
            )

        image_records, num_gt, tp, fp, fn = match_detections(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=sample["boxes"],
            iou_threshold=eval_cfg["iou_threshold"],
        )
        all_records.extend(image_records)
        total_gt += num_gt
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if (index + 1) % 50 == 0 or (index + 1) == len(samples):
            print("已验证: {}/{}".format(index + 1, len(samples)))

    metrics = compute_metrics(
        records=all_records,
        num_gt=total_gt,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
    )

    print("AP@{:.2f}: {:.6f}".format(eval_cfg["iou_threshold"], metrics["ap"]))
    print("Precision: {:.6f}".format(metrics["precision"]))
    print("Recall: {:.6f}".format(metrics["recall"]))
    print("F1-score: {:.6f}".format(metrics["f1"]))
    print("GT 数量: {}".format(metrics["num_gt"]))
    print("预测框数量: {}".format(metrics["num_pred"]))

    save_report(eval_cfg["report_path"], metrics, eval_cfg)
    print("验证报告已保存到: {}".format(eval_cfg["report_path"]))


if __name__ == "__main__":
    main()
