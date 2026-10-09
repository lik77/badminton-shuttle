#!/usr/bin/env python3
# 从录制的 AVI 视频抽帧（供标注/训练用），无需 ffmpeg，用 OpenCV 读。
# 用法:
#   python3 extract_frames.py <视频或目录>... [--out 输出目录] [--stride N]
# 例:
#   python3 extract_frames.py record scene1_xxx --out train/dataset/images/train --stride 10
import sys, os, glob, cv2

def main():
    args = sys.argv[1:]
    out = "frames"
    stride = 10
    inputs = []
    i = 0
    while i < len(args):
        if args[i] == "--out":
            out = args[i + 1]; i += 2
        elif args[i] == "--stride":
            stride = int(args[i + 1]); i += 2
        else:
            inputs.append(args[i]); i += 1

    videos = []
    for inp in inputs:
        if os.path.isdir(inp):
            videos += sorted(glob.glob(os.path.join(inp, "*.avi")))
        else:
            videos.append(inp)

    if not videos:
        print("没找到视频，请检查路径")
        return

    os.makedirs(out, exist_ok=True)
    total = 0
    for v in videos:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            print(f"[跳过] 打不开 {v}")
            continue
        base = os.path.splitext(os.path.basename(v))[0]
        parent = os.path.basename(os.path.dirname(v)) or "v"
        n = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n % stride == 0:
                name = f"{parent}_{base}_{saved:05d}.jpg"
                cv2.imwrite(os.path.join(out, name), frame)
                saved += 1
            n += 1
        cap.release()
        print(f"{v}: 总 {n} 帧 -> 抽 {saved} 张")
        total += saved
    print(f"\n总共抽取 {total} 张 -> {out}/")

if __name__ == "__main__":
    main()
