"""
自建数据集验证配置文件。

使用方式：
1. 直接修改 ACTIVE_VAL_CONFIG 中的路径与参数。
2. 运行 `python val_test/run_val.py` 即可开始验证。

说明：
- `network` 需要与 data/config.py 中已有网络名对应。
- `trained_model` 填写待验证的 pth 权重路径。
- `label_file` 填写验证集 label.txt 路径。
"""

ACTIVE_VAL_CONFIG = {
    # 当前要验证的网络名，必须与训练时使用的网络配置保持一致
    "network": "resnest50",

    # 待验证的模型权重路径
    "trained_model": r"D:\\last_pth\\ResNeSt50_onWiderface.pth",

    # 自建验证集的标注文件路径
    "label_file": r"D:\\cut\\retinaface_dataset\val\\label.txt",

    # 是否使用 CPU 验证；有 GPU 时建议设为 False
    "cpu": False,

    # 预测框置信度阈值
    "confidence_threshold": 0.1,

    # NMS 或 Soft-NMS 阈值
    "nms_threshold": 0.4,

    # NMS 前保留候选框数量
    "top_k": 5000,

    # NMS 后保留候选框数量
    "keep_top_k": 750,

    # 判定 TP 使用的 IoU 阈值
    "iou_threshold": 0.5,

    # 是否输出逐图像预测结果，便于后续可视化与排查
    "save_predictions": False,

    # 预测结果输出目录
    "prediction_dir": r"./val_test/predictions",

    # 最终验证报告输出路径
    "report_path": r"./val_test/val_report.txt",
}


# 可选：给不同实验预留多个配置模板，便于后续切换
VAL_CONFIG_TEMPLATES = {
    "exp01_resnet50_widerface": {
        "network": "resnet50",
        "trained_model": r"./weights/resnet50_final.pth",
        "label_file": r"D:\your_widerface_val\label.txt",
        "cpu": False,
        "confidence_threshold": 0.02,
        "nms_threshold": 0.4,
        "top_k": 5000,
        "keep_top_k": 750,
        "iou_threshold": 0.5,
        "save_predictions": False,
        "prediction_dir": r"./val_test/predictions/exp01_resnet50_widerface",
        "report_path": r"./val_test/reports/exp01_resnet50_widerface.txt",
    },
    "exp02_resnest50_widerface": {
        "network": "resnest50",
        "trained_model": r"./weights/resnest50_widerface_final.pth",
        "label_file": r"D:\your_widerface_val\label.txt",
        "cpu": False,
        "confidence_threshold": 0.02,
        "nms_threshold": 0.4,
        "top_k": 5000,
        "keep_top_k": 750,
        "iou_threshold": 0.5,
        "save_predictions": False,
        "prediction_dir": r"./val_test/predictions/exp02_resnest50_widerface",
        "report_path": r"./val_test/reports/exp02_resnest50_widerface.txt",
    },
    "exp03_resnest50_classroom": {
        "network": "resnest50",
        "trained_model": r"./weights/resnest50_classroom_final.pth",
        "label_file": r"D:\your_dataset\val\label.txt",
        "cpu": False,
        "confidence_threshold": 0.02,
        "nms_threshold": 0.4,
        "top_k": 5000,
        "keep_top_k": 750,
        "iou_threshold": 0.5,
        "save_predictions": False,
        "prediction_dir": r"./val_test/predictions/exp03_resnest50_classroom",
        "report_path": r"./val_test/reports/exp03_resnest50_classroom.txt",
    },
    "exp04_resnest50_p2_classroom": {
        "network": "resnest50_p2",
        "trained_model": r"./weights/resnest50_p2_final.pth",
        "label_file": r"D:\your_dataset\val\label.txt",
        "cpu": False,
        "confidence_threshold": 0.02,
        "nms_threshold": 0.4,
        "top_k": 5000,
        "keep_top_k": 750,
        "iou_threshold": 0.5,
        "save_predictions": False,
        "prediction_dir": r"./val_test/predictions/exp04_resnest50_p2_classroom",
        "report_path": r"./val_test/reports/exp04_resnest50_p2_classroom.txt",
    },
    "exp05_resnest50_atss_classroom": {
        "network": "resnest50_atss",
        "trained_model": r"./weights/resnest50_atss_final.pth",
        "label_file": r"D:\your_dataset\val\label.txt",
        "cpu": False,
        "confidence_threshold": 0.02,
        "nms_threshold": 0.4,
        "top_k": 5000,
        "keep_top_k": 750,
        "iou_threshold": 0.5,
        "save_predictions": False,
        "prediction_dir": r"./val_test/predictions/exp05_resnest50_atss_classroom",
        "report_path": r"./val_test/reports/exp05_resnest50_atss_classroom.txt",
    },
    "exp06_resnest50_p2_se_classroom": {
        "network": "resnest50_p2_se",
        "trained_model": r"./weights/resnest50_p2_se_final.pth",
        "label_file": r"D:\your_dataset\val\label.txt",
        "cpu": False,
        "confidence_threshold": 0.02,
        "nms_threshold": 0.4,
        "top_k": 5000,
        "keep_top_k": 750,
        "iou_threshold": 0.5,
        "save_predictions": False,
        "prediction_dir": r"./val_test/predictions/exp06_resnest50_p2_se_classroom",
        "report_path": r"./val_test/reports/exp06_resnest50_p2_se_classroom.txt",
    },
    "exp07_resnest50_p2_atss_se_classroom": {
        "network": "resnest50_p2_atss_se",
        "trained_model": r"./weights/resnest50_p2_atss_se_final.pth",
        "label_file": r"D:\your_dataset\val\label.txt",
        "cpu": False,
        "confidence_threshold": 0.02,
        "nms_threshold": 0.4,
        "top_k": 5000,
        "keep_top_k": 750,
        "iou_threshold": 0.5,
        "save_predictions": False,
        "prediction_dir": r"./val_test/predictions/exp07_resnest50_p2_atss_se_classroom",
        "report_path": r"./val_test/reports/exp07_resnest50_p2_atss_se_classroom.txt",
    },
    "exp08_resnest50_p2_atss_se_softnms_classroom": {
        "network": "resnest50_p2_atss_se_softnms",
        "trained_model": r"./weights/resnest50_p2_atss_se_softnms_final.pth",
        "label_file": r"D:\your_dataset\val\label.txt",
        "cpu": False,
        "confidence_threshold": 0.02,
        "nms_threshold": 0.4,
        "top_k": 5000,
        "keep_top_k": 750,
        "iou_threshold": 0.5,
        "save_predictions": False,
        "prediction_dir": r"./val_test/predictions/exp08_resnest50_p2_atss_se_softnms_classroom",
        "report_path": r"./val_test/reports/exp08_resnest50_p2_atss_se_softnms_classroom.txt",
    },
}

