# 把 Roboflow 导出的【多边形分割】标签转成【检测包围框】格式
# 输入: class x1 y1 x2 y2 ... (多边形)  或  class cx cy w h (已是框)
# 输出: class cx cy w h  (YOLO 检测格式)
import glob

NL = chr(10)

def convert(pattern):
    cnt = 0
    for f in glob.glob(pattern):
        out = []
        for line in open(f):
            p = line.split()
            if not p:
                continue
            cls = p[0]
            vals = list(map(float, p[1:]))
            if len(vals) == 4:
                out.append("%s %.6f %.6f %.6f %.6f" % (cls, vals[0], vals[1], vals[2], vals[3]))
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                xs = vals[0::2]; ys = vals[1::2]
                x1, x2 = min(xs), max(xs); y1, y2 = min(ys), max(ys)
                cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0
                w = x2 - x1; h = y2 - y1
                out.append("%s %.6f %.6f %.6f %.6f" % (cls, cx, cy, w, h))
            else:
                print("跳过异常行:", f, line.strip())
        open(f, "w").write(NL.join(out) + (NL if out else ""))
        cnt += 1
    return cnt

n = (convert("roboflow/train/labels/*.txt") +
     convert("roboflow/valid/labels/*.txt") +
     convert("roboflow/test/labels/*.txt"))
print("转换标签文件数:", n)
