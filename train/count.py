import glob
for name in ["roboflow", "shuttlecock"]:
    files = glob.glob(name + "/*/labels/*.txt")
    instances = 0
    nonempty = 0
    for f in files:
        n = 0
        for line in open(f, encoding="utf-8", errors="ignore"):
            if line.strip():
                n += 1
        if n:
            nonempty += 1
            instances += n
    print("%s: 标签文件=%d  有目标的帧=%d  实例(框)总数=%d" % (name, len(files), nonempty, instances))
