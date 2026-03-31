from __future__ import print_function
import argparse
import cv2
import torch
#"D:\last_pth\ResNeSt50_P2_Final.pth"
#python eval_custom_dataset.py --network resnest50_p2_se --trained_model ./weights/你的模型.pth --label_file D:\你的验证集\label.txt
#python eval_custom_dataset.py --network resnest50 --trained_model D:/last_pth/ResNeSt50_P2_Final.pth --label_file D:/cut/retinaface_dataset/val/label.txt

from data import cfg_mnet, cfg_re50, cfg_resnest50, cfg_re50_p2, cfg_resnest50_p2, cfg_re50_p2_se, cfg_resnest50_p2_se
from models.retinaface import RetinaFace
from utils.custom_eval import parse_retinaface_label_file, run_detector, match_detections, compute_metrics


parser = argparse.ArgumentParser(description='RetinaFace 自定义数据集评估')
parser.add_argument('-m', '--trained_model', required=True, type=str, help='待评估的模型权重路径')
parser.add_argument('--network', required=True, help='网络配置名，如 resnest50_p2_se')
parser.add_argument('--label_file', required=True, type=str, help='验证集 label.txt 路径')
parser.add_argument('--cpu', action="store_true", default=False, help='是否使用 CPU 推理')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='置信度阈值')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='NMS 或 Soft-NMS 阈值')
parser.add_argument('--top_k', default=5000, type=int, help='NMS 前保留的候选框数量')
parser.add_argument('--keep_top_k', default=750, type=int, help='NMS 后保留的候选框数量')
parser.add_argument('--iou_threshold', default=0.5, type=float, help='判定 TP 的 IoU 阈值')
args = parser.parse_args()


def get_cfg(network_name):
    if network_name == "mobile0.25":
        return cfg_mnet
    if network_name == "resnet50":
        return cfg_re50
    if network_name == "resnest50":
        return cfg_resnest50
    if network_name == "resnet50_p2":
        return cfg_re50_p2
    if network_name == "resnest50_p2":
        return cfg_resnest50_p2
    if network_name == "resnet50_p2_se":
        return cfg_re50_p2_se
    if network_name == "resnest50_p2_se":
        return cfg_resnest50_p2_se
    raise ValueError("Unsupported network: {}".format(network_name))


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = get_cfg(args.network)
    device = torch.device("cpu" if args.cpu else "cuda")

    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    net = net.to(device)

    samples = parse_retinaface_label_file(args.label_file)
    print('验证样本数: {}'.format(len(samples)))

    all_records = []
    total_gt = 0

    for index, sample in enumerate(samples):
        image_bgr = cv2.imread(sample['image_path'])
        if image_bgr is None:
            raise ValueError('图像读取失败: {}'.format(sample['image_path']))

        pred_boxes, pred_scores = run_detector(
            net=net,
            cfg=cfg,
            image_bgr=image_bgr,
            device=device,
            confidence_threshold=args.confidence_threshold,
            nms_threshold=args.nms_threshold,
            top_k=args.top_k,
            keep_top_k=args.keep_top_k
        )

        image_records, num_gt = match_detections(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            gt_boxes=sample['boxes'],
            iou_threshold=args.iou_threshold
        )
        all_records.extend(image_records)
        total_gt += num_gt

        if (index + 1) % 50 == 0 or (index + 1) == len(samples):
            print('已评估: {}/{}'.format(index + 1, len(samples)))

    metrics = compute_metrics(all_records, total_gt)
    print('AP@{:.2f}: {:.6f}'.format(args.iou_threshold, metrics['ap']))
    print('Precision: {:.6f}'.format(metrics['precision']))
    print('Recall: {:.6f}'.format(metrics['recall']))
    print('F1-score: {:.6f}'.format(metrics['f1']))
    print('GT 数量: {}'.format(metrics['num_gt']))
    print('预测框数量: {}'.format(metrics['num_pred']))
