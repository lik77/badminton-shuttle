#!/usr/bin/env python3
# 两组外参标定：算 tele(16mm) 光轴系 → wide(8mm) 光轴系 的 R_t2w / t_t2w。
#
# 前提：先分别跑完 calibrate_stereo.py wide / tele（得到 stereo_wide.npz、stereo_tele.npz）。
# 输入：两组【同时】拍到同一块棋盘格的图片（硬触发同帧）：
#        wide_left/*.png 与 tele_left/*.png 同名成对。
# 输出：打印 R_t2w / t_t2w，并自动写回 ../configs/rigs.yaml。
#
# 原理：对同一帧，分别在 wide 左相机、tele 左相机里 solvePnP 得到棋盘格位姿，
#       再由 p_wide = R_t2w * p_tele + t_t2w 解出两组相对位姿，多帧取平均。
import sys, glob, os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))   # .../shuttle/calibration
ROOT = os.path.dirname(HERE)                        # .../shuttle

CHECKERS = [(11, 8), (8, 11)]; SQUARE = 0.015
W_DIR = os.path.join(HERE, "wide_left", "*.png"); T_DIR = os.path.join(HERE, "tele_left", "*.png")
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

def _find_corners(img):
    for (cw, ch) in CHECKERS:
        ok, c = cv2.findChessboardCorners(img, (cw, ch), None)
        if ok:
            return ok, c, (cw, ch)
    return False, None, None

def _objp(cw, ch):
    o = np.zeros((cw * ch, 3), np.float32)
    o[:, :2] = np.mgrid[0:cw, 0:ch].T.reshape(-1, 2) * SQUARE
    return o

w = np.load(os.path.join(HERE, "stereo_wide.npz")); t = np.load(os.path.join(HERE, "stereo_tele.npz"))
KW, dW = w["KL"], w["distL"]; KT, dT = t["KL"], t["distL"]

wf = sorted(glob.glob(W_DIR)); tf = sorted(glob.glob(T_DIR))
assert len(wf) == len(tf) and len(wf) >= 5, "两组图片要同名成对且 >=5 张"

Rt2w_list, tt2w_list = [], []
for a, b in zip(wf, tf):
    W = cv2.imread(a, 0); T = cv2.imread(b, 0)
    okW, cW, sW = _find_corners(W)
    okT, cT, sT = _find_corners(T)
    if not (okW and okT): continue
    if sW != sT: continue
    cW = cv2.cornerSubPix(W, cW, (11,11), (-1,-1), criteria)
    cT = cv2.cornerSubPix(T, cT, (11,11), (-1,-1), criteria)
    _, rW, tW = cv2.solvePnP(_objp(*sW), cW, KW, dW)
    _, rT, tT = cv2.solvePnP(_objp(*sT), cT, KT, dT)
    RW, _ = cv2.Rodrigues(rW); RT, _ = cv2.Rodrigues(rT)
    R = RW @ RT.T
    tv = tW.ravel() - R @ tT.ravel()
    Rt2w_list.append(R); tt2w_list.append(tv)

R = np.mean(np.stack(Rt2w_list), axis=0)
# 重新正交化（平均后的旋转矩阵未必严格正交）
u, _, vt = np.linalg.svd(R); R = u @ vt
tvec = np.mean(np.stack(tt2w_list), axis=0)

print("R_t2w =")
print(np.array2string(R, precision=5, suppress_small=True))
print("t_t2w =", np.array2string(tvec, precision=5, suppress_small=True))
print("R_t2w 展平(行主序):", [f"{x:.5f}" for x in R.ravel()])
print("t_t2w:", [f"{x:.5f}" for x in tvec])

# 写回 rigs.yaml
path = os.path.join(ROOT, "configs", "rigs.yaml")
if os.path.exists(path):
    s = open(path).read()
    s = s.replace("R_t2w: [1, 0, 0,  0, 1, 0,  0, 0, 1]",
                  "R_t2w: [" + ", ".join(f"{x:.5f}" for x in R.ravel()) + "]")
    s = s.replace("t_t2w: [0.0, 0.0, 0.0]",
                  "t_t2w: [" + ", ".join(f"{x:.5f}" for x in tvec) + "]")
    open(path, "w").write(s)
    print(f"已写回 {path}")
