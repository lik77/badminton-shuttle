# 白羽毛球感知系统（shuttle）—— 4 相机 / 两套双目

针对"4 台海康相机（2×8mm 宽视场 + 2×16mm 长焦）→ 两套双目 → 远/近交接 → 拦截点预测"。

## 1. 架构与数据流

    硬触发(MCU) ──> 4 相机同步曝光
      ├─ 8mm 组 (wide)：宽视场，全程粗定位 + 近场高频精轨迹
      └─ 16mm 组(tele)：长焦，预瞄发射区，远场精轨迹

    未锁定：wide 全帧粗定位 → reset EKF
    已锁定：按距离切换主组
        > handoff_dist(4m)  → tele 为主（远场精轨迹），wide 兜底
        ≤ handoff_dist      → wide 为主（近场高频），tele 兜底
    两组测量统一到世界系(+X前/+Y左/+Z上) → 同一个 DragPredictor 融合
    输出 intercept(t_hit, pos, vel, confidence)

## 2. 依赖 & 编译
    sudo apt install -y libopencv-dev libeigen3-dev libyaml-cpp-dev
    cmake -B build && make -C build -j$(nproc)

## 3. 标定（三步）
### ① 每组双目标定（跑两次）
    cd calibration
    # 8mm 组：左右成对棋盘格放 left/ right/
    python calibrate_stereo.py wide
    # 16mm 组：换图片再跑
    python calibrate_stereo.py tele
    # 产出 stereo_maps_<name>.xml + configs/stereo_<name>.yaml + stereo_<name>.npz

### ② 两组外参（tele → wide）
    # 两组同时拍到同一棋盘格，图片放 wide_left/ 与 tele_left/ 同名成对
    python calibrate_rig_extrinsics.py
    # 自动写回 ../configs/rigs.yaml 的 R_t2w / t_t2w

### ③ （以后）手眼标定
    相机装到机器人上后，标相机→身体外参，修正 configs/rigs.yaml 的 R_c2w。

## 4. 相机 & 模型配置
- 改 configs/rigs.yaml：两组序列号、曝光/增益、外参、handoff_dist。
- YOLO26 单类 shuttle 导出 ONNX，放运行目录，改 src/main.cpp 的 kYoloOnnx。

## 5. 运行
    ./build/shuttle

## 6. 调参（改 config 免重编译）
- configs/shuttle.yaml       检测阈值（亮度/尺寸/圆度/运动外扩/提议周期）
- configs/drag_predictor.yaml EKF（sigma_accel/sigma_pos/sigma_range/k_init/nis_threshold/hit_height）
- configs/stereo_<name>.yaml 双目（f_px/cx/cy/baseline，标定自动生成）
- configs/rigs.yaml          handoff_dist（远/近切换距离）

## 7. 文件结构
    src/hik_camera.*       单相机封装（MVS：开/配/抓/设备时间戳）
    src/stereo_camera.*    一个双目组：同步帧对 + t_capture
    src/rigs.*             两套双目管理 + 外参 + 坐标系统一
    src/yolo_proposer.*     YOLO26 ONNX 候选框（OpenCV DNN）
    src/shuttle_detector.*  框内经典精修（阈值+轮廓+尺寸/圆度+亚像素质心）
    src/stereo.*            三角化 + 右图匹配
    src/rectifier.*         立体校正
    src/drag_predictor.*    阻力模型 EKF + NIS 门控 + 时间戳回放 + 拦截点
    tools/extended_kalman_filter.*  泛型 EKF

## 8. 已知待办（按优先级）
1. R_c2w 用真实手眼标定/IMU 修正（现在是"相机水平前视"近似）。
2. 相机动起来后的自运动补偿（IMU + 前向运动学）。
3. sigma_pos 随距离 z²/(f·b) 自适应。
4. 发射瞬间兜底（YOLO 抓不到时加背景减除+运动预筛）。
