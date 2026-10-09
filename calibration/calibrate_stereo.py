#!/usr/bin/env python3
# 双目立体标定脚本（每组跑一次）
# 用法：
#   python calibrate_stereo.py wide    # 8mm 组：图片放 left/ right/
#   python calibrate_stereo.py tele    # 16mm 组：图片放 left/ right/
# 输出：
#   stereo_maps_<name>.xml       —— C++ StereoRectifier 用
#   configs/stereo_<name>.yaml   —— f_px/cx/cy/baseline（C++ StereoRig 用）
#   stereo_<name>.npz            —— 内参/外参/校正映射，供跨组标定脚本用
#
# 拍摄要点：两台相机刚性固定、锁焦，硬触发同时曝光，棋盘格拍四角/远近 ≥15 对。
import os
import sys
import glob
import numpy as np
import cv2

# 路径全部锚定到脚本自身位置，跟「从哪里运行」无关
HERE = os.path.dirname(os.path.abspath(__file__))   # .../shuttle/calibration
ROOT = os.path.dirname(HERE)                        # .../shuttle

NAME = sys.argv[1] if len(sys.argv) > 1 else "wide"

CHECKERS = [(11, 8), (8, 11)]   # 棋盘格内角点 列×行（横竖两个方向都试）
SQUARE = 0.015                 # 每格边长(米)
MAX_PAIRS = 40                 # 最多用多少对参与标定（多了冗余、还慢）

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

def _imgs(sub):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    base = os.path.join(HERE, sub)
    return sorted([p for e in exts for p in glob.glob(os.path.join(base, "*" + e))])

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

lf = _imgs("left"); rf = _imgs("right")
assert len(lf) == len(rf), "左右图片数不一致"

op, ipL, ipR = [], [], []
W = H = None
for a, b in zip(lf, rf):
    L = cv2.imread(a, 0); R = cv2.imread(b, 0)
    if W is None: H, W = L.shape
    okL, cL, sL = _find_corners(L)
    okR, cR, sR = _find_corners(R)
    if not (okL and okR): continue
    if sL != sR: continue   # 左右方向不一致（理论上不会），跳过
    cL = cv2.cornerSubPix(L, cL, (11, 11), (-1, -1), criteria)
    cR = cv2.cornerSubPix(R, cR, (11, 11), (-1, -1), criteria)
    op.append(_objp(*sL)); ipL.append(cL); ipR.append(cR)
print(f"共 {len(lf)} 对图，检测到角点并参与标定 {len(op)} 对")
if len(op) > MAX_PAIRS:
    step = len(op) / MAX_PAIRS
    sel = [int(i * step) for i in range(MAX_PAIRS)]
    op, ipL, ipR = [op[i] for i in sel], [ipL[i] for i in sel], [ipR[i] for i in sel]
    print(f"图太多，均匀抽取 {MAX_PAIRS} 对参与标定（省时、精度不损失）")
assert len(op) >= 10, (f"有效图对不足 10（只有 {len(op)} 对）。\n"
                          "检查：棋盘格角点数(11×8)是否对、远近/反光/清晰度，"
                          "并确认 CHECKER/SQUARE 与实际板子一致。")

_, KL, distL, _, _ = cv2.calibrateCamera(op, ipL, (W, H), None, None)
_, KR, distR, _, _ = cv2.calibrateCamera(op, ipR, (W, H), None, None)
ret, KL, distL, KR, distR, R, T, E, F = cv2.stereoCalibrate(
    op, ipL, ipR, KL, distL, KR, distR, (W, H), criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC)
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(KL, distL, KR, distR, (W, H), R, T, alpha=0)
mapl1, mapl2 = cv2.initUndistortRectifyMap(KL, distL, R1, P1, (W, H), cv2.CV_32FC1)
mapr1, mapr2 = cv2.initUndistortRectifyMap(KR, distR, R2, P2, (W, H), cv2.CV_32FC1)

np.savez(os.path.join(HERE, f"stereo_{NAME}.npz"), KL=KL, distL=distL, KR=KR, distR=distR,
         R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         mapl1=mapl1, mapl2=mapl2, mapr1=mapr1, mapr2=mapr2)

fs = cv2.FileStorage(os.path.join(ROOT, f"stereo_maps_{NAME}.xml"), cv2.FILE_STORAGE_WRITE)
fs.write("mapl1", mapl1); fs.write("mapl2", mapl2)
fs.write("mapr1", mapr1); fs.write("mapr2", mapr2); fs.release()

with open(os.path.join(ROOT, "configs", f"stereo_{NAME}.yaml"), "w") as f:
    f.write(f"f_px: {P1[0,0]:.3f}\n")
    f.write(f"cx: {P1[0,2]:.3f}\n")
    f.write(f"cy: {P1[1,2]:.3f}\n")
    f.write(f"baseline: {np.linalg.norm(T):.4f}\n")
    f.write("min_disparity: 1.0\n")
    f.write("match_search_px: 40.0\n")

print(f"[{NAME}] RMS={ret:.4f}  f_px={P1[0,0]:.2f}  baseline={np.linalg.norm(T):.4f}m")
print(f"[{NAME}] 已写 stereo_maps_{NAME}.xml / configs/stereo_{NAME}.yaml / stereo_{NAME}.npz")
