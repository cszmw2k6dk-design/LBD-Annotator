# yolo26 —— LBD 标注数据的训练/评估工程

配套 `lbd_annotator.py` 用的训练工具链：把 X-AnyLabeling 标注整理成 YOLO 数据集、
训练、评估、生成补标建议。

> **数据不入库**：`dataset/`、`runs/`、`models/`、`补标建议/`、`.venv/`
> 都在 `.gitignore` 里。这里只放脚本 —— 图纸和标注属于客户资料。

## 脚本一览

| 脚本 | 干什么 |
| --- | --- |
| `scan_dataset.py` | 扫描各批标注目录，统计图片/标注数量、类别分布、跨目录重名 |
| `prepare_yolo.py` | 生成 YOLO 数据集：同名成对、按图纸分组切 train/val、大图缩到统一长边 |
| `verify_dataset.py` | 抽查数据集：把 labels 画回图上确认对齐、统计尺寸分布 |
| `preview_boxes.py` | 把某个 json 的框画到图上，肉眼确认标注对象和尺度 |
| `train_yolo26.py` | 训练入口（CPU 也能跑）。支持 `lr0` 等参数透传、`--resume` 续训 |
| `progress.py` | 一行命令看训练进度 / ETA / 最近指标（`查看进度.bat` 双击版） |
| `eval_model.py` | 单独跑一次验证，拿分类别的 P/R/mAP（训练中不打印 per-class） |
| `practical_eval.py` | 按真实用法评估：各置信度阈值下的 P/R/F1、可开 TTA 对比 |
| `export_model.py` | 把权重导出到 `models/<名字>/`，附 `classes.txt` 和 `meta.txt`（记 imgsz） |
| `get_weights.py` | 下载 yolo26n/s 预训练权重 |
| `watch_and_continue.py` / `watch_chain_2560.py` | 看门狗：等上一轮跑完自动接下一轮 |
| `hard_examples.py` | 难例清单：找"模型和标注分歧最大"的图（多半是漏标/错标） |
| `annot_offset.py` | 标注偏移清单：按"框边离图纸边界的距离"排序，抽查用 |
| `export_missing.py` | 把"模型找到、标注没有"的框导出成 X-AnyLabeling 文件（补标用） |
| `merge_suggest.py` | 把审完的补标建议并回原始标注，输出到 `补标结果/` |
| `diag_*.py` | 各种诊断：输入尺度、边缘吸附、框大小偏置、形状过滤等 |
| `test_clean.py` / `test_snap.py` | 自检：标注清理规则、边缘吸附实现 |

## 典型流程

```bat
:: 1) 整理数据集（默认长边 2560，大图用 LANCZOS 缩，小图原样）
python prepare_yolo.py --max-side 2560

:: 2) 抽查数据集对不对
python verify_dataset.py --n 4

:: 3) 训练
python train_yolo26.py --imgsz 2560 --epochs 25 --batch 4

:: 4) 看进度 / 评估 / 导出
python progress.py
python eval_model.py --imgsz 2560
python practical_eval.py --imgsz 2560
python export_model.py --name <那轮的名字>
```

## 几条踩过的坑（写在这儿免得再踩）

- **训练/推理的预处理必须一致**。训练时用 LANCZOS 缩到 2560，推理时如果直接把
  9000×6000 丢给 YOLO，内部会用 cv2 的一步线性插值降到 imgsz，细线会被"跳过"
  （实测 Tracker 召回 0.68→0.44）。所以 `lbd_annotator.py` 里加了
  `model_input_image()`，喂模型前先平滑缩放。
- **`optimizer=auto` 会忽略 `lr0`**。想显式控制学习率就得把优化器也定死，
  `train_yolo26.py` 里已经自动处理。
- **`resume=True` 改不了 epochs**，它只是接着跑完原来的轮数。要"加训"得从
  `best.pt` 重新起一轮微调。
- **去重的重叠门槛别设太高**。同一个物体被两次识别画出来，重叠常落在
  0.8~0.9，门槛设 0.9 会漏掉近三成的重复框。
- **eval 的分类别指标只在训练收尾那次验证才打印**，训练中要看就用 `eval_model.py`。
