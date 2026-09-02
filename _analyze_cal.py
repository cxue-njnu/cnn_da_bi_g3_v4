# -*- coding: utf-8 -*-
import glob, json, os, numpy as np
base = r"E:\seizure pred\cnn_da_bi_g3_v4\runs\seed42\results"
files = [f for f in sorted(glob.glob(base + "/CNN_DA_BI_G3_V4_chb08_event*.json")) if f.endswith(".done.json") is False]
print("fold npos nneg sharp pos_pmean neg_pmean rawlogit_rms dir_disc_mean")
for f in files:
    d = json.load(open(f)); fold = os.path.basename(f).replace(".json","")
    y = np.array(d["test_labels"]); s = np.array(d["test_logits"]); rl = np.array(d["test_raw_logits"])
    dd = np.array(d.get("direction_disagreement", []))
    npos = int((y==1).sum()); nneg = int((y==0).sum())
    sharp = float(np.mean((s < 0.01) | (s > 0.99)))
    pos_p = float(s[y==1].mean()) if npos else float("nan")
    neg_p = float(s[y==0].mean()) if nneg else float("nan")
    rms = float(np.sqrt((rl**2).mean()))
    dd_m = float(dd.mean()) if len(dd) else float("nan")
    print("%s %d %d %.3f %.3f %.3f %.2f %.3f" % (fold.split("_")[-1], npos, nneg, sharp, pos_p, neg_p, rms, dd_m))