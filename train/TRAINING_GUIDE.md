# 羽毛球 YOLO26 训练 / 导出 / 测试 说明文档

这份文档解释每一步的命令和参数，覆盖从数据到部署的完整流程。

---

## 0. 整体流程

    训练(数据集+命令) → best.pt → 导出 ONNX → 拷进 shuttle/ → C++ 部署

两条线要分清：
- 训练/导出/测试：Python + ultralytics，在 ~/yolo_env 环境里跑（本文档）
- 部署：C++ shuttle/ 工程，加载 ONNX 做检测（见 shuttle/README.md）

---

## 1. 环境

激活虚拟环境（每次新开终端都要）：

    source ~/yolo_env/bin/activate

验证 GPU 可用：

    python -c "import torch; print(torch.cuda.is_available())"   # 应打印 True

---

## 2. 数据集

位置：shuttle/train/roboflow/

    roboflow/
      train/images/*.jpg   训练图
      train/labels/*.txt   对应标签
      valid/images, valid/labels   验证集
      test/images, test/labels     测试集

标签格式（YOLO 检测框，每行一个框）：

    class cx cy w h     # 全部归一化 0~1；单类 class 恒为 0

数据集配置 data.yaml（已建好）：

    path: /home/liki/Badminton Perception/shuttle/train/roboflow
    train: train/images
    val: valid/images
    test: test/images
    nc: 1
    names: ['shuttle']

---

## 3. 训练命令（逐参数解释）

    yolo detect train \
      model=yolo26n.pt \
      data="/home/liki/Badminton Perception/shuttle/train/data.yaml" \
      epochs=100 imgsz=1280 batch=8 device=0

参数说明：

| 参数 | 含义 | 建议值 |
| --- | --- | --- |
| model | 起点权重，yolo26n.pt=YOLO26-nano 预训练 | 从头训用 yolo26n.pt；微调用上次 best.pt |
| data | 数据集配置 yaml 路径 | 上面的 data.yaml |
| epochs | 训练轮数 | 100~200 |
| imgsz | 输入分辨率 | 小球用 1280；640 更快但漏小球 |
| batch | 批大小 | 8GB 显存 1280 用 4~8；OOM 就降 |
| device | 用哪张卡 | 0=第一张 GPU；cpu=用 CPU |
| lr0 | 初始学习率 | 0.005~0.01 |
| optimizer | 优化器 | auto/SGD/Adam |
| patience | 早停：N 轮不涨就停 | 50 |
| workers | 数据加载线程 | 4~8，内存紧张调小 |
| fraction | 只用数据的多少比例 | 冒烟测试用 0.1 |
| cache | 缓存方式 | disk=缓存到磁盘(省CPU)；ram=内存(吃内存) |
| amp | 混合精度 | true 默认，省显存加快 |
| mosaic | 马赛克增强 | 1.0 默认开 |

训练产出：runs/detect/train-N/weights/best.pt（N 每次递增）。

---

## 4. 评估指标（训练完自动算）

训练日志和 runs/detect/train-N/ 里：

| 指标 | 含义 |
| --- | --- |
| P (Precision) | 检出框里有多少是真球 |
| R (Recall) | 真球有多少被检出 |
| mAP50 | IoU=0.5 下平均精度，最常用的好坏指标 |
| mAP50-95 | IoU 0.5~0.95 平均，更严格 |

查看：

    cat runs/detect/train-N/results.csv   # 每轮数值
    # 还有 PR_curve.png / confusion_matrix.png / F1_curve.png 等图

---

## 5. 导出 ONNX

    yolo export model="runs/detect/train-N/weights/best.pt" format=onnx imgsz=1280

产出 best.onnx。输出 shape 应为 [1, 5, N]（单类：4框+1置信度），
与 shuttle 工程 yolo_proposer.cpp 的解码一致。

---

## 6. 测试

视频/图片/摄像头：

    yolo predict model="runs/detect/train-N/weights/best.pt" \
      source="/home/liki/视频/badminton.mp4" save=True conf=0.25 vid_stride=3

参数：

| 参数 | 含义 |
| --- | --- |
| source | 输入：视频路径 / 图片路径 / 0=摄像头 |
| save | true=存画框结果到 runs/detect/predict-N/ |
| show | true=弹窗实时显示 |
| conf | 置信度阈值，越高越严、越低越多 |
| vid_stride | 每 N 帧推理一次（加速测试） |

---

## 7. 接 shuttle 部署

    1. 导出 ONNX（见第 5 节）
    2. cp best.onnx → shuttle/shuttle_yolo26.onnx
    3. 改 shuttle/src/main.cpp 的 kYoloOnnx 路径（默认已是 shuttle_yolo26.onnx）
    4. cd shuttle && cmake -B build && make -C build -j

---

## 8. 常见问题

1) 训练爆内存/卡
   根因：内存不够换页。解决：关浏览器腾内存、batch=4 workers=4、
   冒烟测试加 fraction=0.1；实在不够才降 imgsz=960（会损失小球精度）。

2) 推理慢
   测试阶段：vid_stride=3 跳帧；用 640 模型更快。
   部署阶段：shuttle 工程用 ROI 追踪，不逐帧全图，天然快。

3) 框不准（框错/框大/漏框）
   根因：训练数据问题。修法：只用真实的小目标追踪数据、imgsz=1280、训满 epoch；
   最终要用自己相机拍的白球图 fine-tune。

---

## 9. 待办/下一步

- 用真实数据（相机白球图 / 比赛视频抽帧）标注并 fine-tune；
- 正式部署：ONNX 接 shuttle 工程，双目相机 7~8m 白球实测。

