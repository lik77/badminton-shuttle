#!/usr/bin/env python3
# 把 Roboflow 导出的 COCO 格式数据集转成 YOLO 格式（ultralytics 训练用）。
# 用法: python3 convert_coco_to_yolo.py <COCO数据集目录> <输出目录>
import json, os, shutil, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/liki/下载/9.23badminton"
DST = sys.argv[2] if len(sys.argv) > 2 else "/home/liki/Badminton Perception/shuttle/train/dataset"

total_img = total_box = 0
for split in ["train", "valid", "test"]:
    coco_path = os.path.join(SRC, split, "_annotations.coco.json")
    if not os.path.exists(coco_path):
        print(f"[跳过] 没有 {coco_path}")
        continue
    coco = json.load(open(coco_path))
    img_info = {img["id"]: img for img in coco["images"]}

    # 只要 badminton 类（category_id==1），映射为 YOLO class 0
    ann_by_img = {}
    for ann in coco["annotations"]:
        if ann.get("category_id") == 1:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

    dst_img = os.path.join(DST, "images", split)
    dst_lbl = os.path.join(DST, "labels", split)
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    n_img = n_box = 0
    for img_id, info in img_info.items():
        fname = info["file_name"]
        W, H = info["width"], info["height"]
        # 用原始文件名（extra.name），去掉扩展名，避免 Roboflow 的 .rf.hash
        name = info.get("extra", {}).get("name", fname)
        base = name.rsplit(".", 1)[0]
        shutil.copy(os.path.join(SRC, split, fname), os.path.join(dst_img, base + ".jpg"))

        lines = []
        for ann in ann_by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_box += 1
        open(os.path.join(dst_lbl, base + ".txt"), "w").write("\n".join(lines))
        n_img += 1

    print(f"{split}: {n_img} 图, {n_box} 框")
    total_img += n_img
    total_box += n_box

print(f"\n完成: 共 {total_img} 图, {total_box} 框 -> {DST}/")
