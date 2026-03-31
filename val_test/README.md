# 自建数据集验证模块说明

## 1. 模块用途

本目录用于对自建课堂人脸数据集的验证集进行本地评估，并输出以下指标：

- `AP@0.5`
- `Precision`
- `Recall`
- `F1-score`

该模块的设计目的有两个：

1. 便于在本地直接对训练得到的 `pth` 权重做验证。
2. 便于在论文中说明“自建数据集验证过程”以及“AP 等指标的获得方式”。`r`n3. 统一管理 8 组实验的验证命名，避免训练名、验证名和论文表格名称不一致。

---

## 2. 文件说明

- [custom_val_config.py](D:\Retinaface\val_test\custom_val_config.py)
  自建验证配置文件，类似训练的 `config`，方便直接改路径和参数。

- [eval_utils.py](D:\Retinaface\val_test\eval_utils.py)
  验证过程中使用的工具函数，包括：
  - 标注解析
  - 图像预处理
  - 单图检测
  - IoU 计算
  - AP / Precision / Recall / F1 计算
  - 预测结果与报告保存

- [run_val.py](D:\Retinaface\val_test\run_val.py)
  验证入口脚本。运行后可直接输出最终验证指标。

---

## 3. 运行方式

先修改 [custom_val_config.py](D:\Retinaface\val_test\custom_val_config.py) 中的以下字段：

- `network`
- `trained_model`
- `label_file`
- `report_path`

然后运行：

```bash
python val_test/run_val.py
```

如果你使用的是 `yolo11` 环境，也可以直接用环境解释器运行：

```bash
D:\anaconda\envs\yolo11\python.exe val_test/run_val.py
```

---

## 4. 验证流程说明

本模块的验证流程如下：

1. 读取验证集 `label.txt`
2. 加载训练完成的 `pth` 模型权重
3. 对验证集中的每张图片执行前向推理
4. 结合配置中的 `confidence_threshold` 与 `nms_threshold` 完成后处理
5. 将预测框与真实标注框按 `IoU` 阈值进行匹配
6. 统计 `TP`、`FP`、`FN`
7. 计算 `AP@0.5`、`Precision`、`Recall`、`F1-score`
8. 将结果保存为验证报告

---

## 5. 论文写作建议

建议你在论文第四章“实验设计与结果分析”中增加一小段“自建数据集验证模块说明”，可参考下面这段写法：

> 为了对自建课堂人脸数据集上的模型性能进行定量分析，本文设计并实现了独立的验证模块。该模块首先读取验证集标注文件与训练完成的模型权重，然后对验证集中每张图像执行前向推理，并在后处理后得到预测框结果。进一步地，将预测框与真实标注框按照 IoU 阈值进行匹配，统计真正例、假正例与假负例数量，最终计算 AP@0.5、Precision、Recall 和 F1-score 等指标。该模块保证了自建数据集实验结果的可重复性，也为论文中的性能对比分析提供了统一评价标准。

如果你愿意，还可以在第四章补一个小节：

- `4.X 自建数据集验证模块设计`

这样论文中的指标来源会更自然，也更完整。

---

## 6. 8 组实验命名建议

你当前 8 组实验建议统一采用以下命名：

1. `resnet50`
2. `resnest50`
3. `resnest50`
4. `resnest50_p2`
5. `resnest50_atss`
6. `resnest50_p2_se`
7. `resnest50_p2_atss_se`
8. `resnest50_p2_atss_se_softnms`

其中第 2 组和第 3 组网络结构相同，只是训练数据集不同，因此论文中应说明“该组实验仅改变数据集来源，其他训练参数保持一致”。

---

## 7. 控制变量说明

`config` 的配置取决于你的实验目的，而不是随意修改。做对照实验时，核心原则是“只改一个因素，其余条件保持一致”。

建议按下面的思路控制变量：

1. 比较主干网络时：
   - 只改 `ResNet50` 与 `ResNeSt50`
   - 其余训练参数保持一致

2. 比较 P2 时：
   - 只改 `P2` 是否加入
   - 不改 `ATSS`、`SE`、`Soft-NMS`

3. 比较 ATSS + 损失函数替换时：
   - 只改 `use_atss`
   - 其余结构不动

4. 比较 SSH 注意力时：
   - 只改 `ssh_type`
   - 其余保持一致

5. 比较 Soft-NMS 时：
   - 只改 `use_soft_nms`
   - 其余验证阈值保持不变

训练阶段建议固定不变的参数包括：

- `batch_size`
- `epoch`
- `decay1`
- `decay2`
- `image_size`
- `optimizer`
- `learning rate`

验证阶段建议固定不变的参数包括：

- `confidence_threshold`
- `nms_threshold`
- `iou_threshold`
- `top_k`
- `keep_top_k`

简单说：

- 训练配置决定“模型学什么”
- 验证配置决定“模型怎么评”

如果你在验证时频繁改阈值，那么不同实验之间就失去了公平对照意义。
