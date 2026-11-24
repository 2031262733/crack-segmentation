# -*- coding: utf-8 -*-
"""Generate result figures for the repo: qualitative comparison, threshold sweep, training curves."""
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset import CrackDataset
from models import build_model

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

SIZE = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 1. 定性对比 ----------------
ds = CrackDataset("data/cfd", "test", SIZE, train=False)
pick, wanted = [], [("CFD", False), ("CRACK500", False), ("Volker", False)]
for i in range(len(ds)):
    stem = Path(ds.images[i]).stem
    for w, kw in enumerate(wanted):
        if not kw[1] and stem.startswith(wanted[w][0]):
            gt = np.array(Image.open(ds.masks[i]).convert("L"))
            if (gt > 127).sum() > 400:
                pick.append(i)
                wanted[w] = (wanted[w][0], True)
    if len(pick) == 3:
        break
print("picked", [Path(ds.images[i]).stem for i in pick])

models = {}
for name, ckpt in (("U-Net", "runs/unet/best.pt"), ("DeepCrack", "runs/deepcrack/best.pt")):
    ck = torch.load(ckpt, map_location=device, weights_only=True)
    m = build_model(ck["model"], pretrained=False).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    models[name] = m

imgs, gts, preds = [], [], {k: [] for k in models}
with torch.no_grad():
    for i in pick:
        x, y = ds[i]
        imgs.append(x)
        gts.append((y[0].numpy() > 0.5))
        for k, m in models.items():
            p = torch.sigmoid(m(x.unsqueeze(0).to(device))[-1])[0, 0].cpu().numpy()
            preds[k].append(p > 0.5)

def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return 1.0 if union == 0 else inter / union

def err_panel(img, g, p):
    """灰底 + TP 绿 / FP 红 / FN 蓝"""
    canvas = (0.55 * img + 0.45 * 1.0)
    canvas = np.stack([canvas] * 3, -1)
    canvas[g & p] = (0.30 * canvas[g & p] + 0.70 * np.array([0.13, 0.75, 0.35]))
    canvas[p & ~g] = (0.30 * canvas[p & ~g] + 0.70 * np.array([0.85, 0.20, 0.20]))
    canvas[g & ~p] = (0.30 * canvas[g & ~p] + 0.70 * np.array([0.15, 0.40, 0.85]))
    return np.clip(canvas, 0, 1)

fig, axes = plt.subplots(len(pick), 4, figsize=(6.9, 5.1))
for r, i in enumerate(range(len(pick))):
    img = imgs[i].permute(1, 2, 0).numpy()
    img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    g = gts[i]
    cells = [(img, "输入"), (np.stack([g] * 3, -1).astype(float), "人工标注")]
    for k in models:
        p = preds[k][i]
        cells.append((err_panel(img.mean(2), g, p), k))
    for c, (arr, title) in enumerate(cells):
        ax = axes[r, c]
        ax.imshow(arr)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if r == 0:
            ax.set_title(title, fontsize=8, pad=3)
        if c >= 2:
            ax.text(0.97, 0.04, f"IoU {iou(g, preds[title][i]):.3f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6.6, color="w",
                    bbox=dict(fc="0.1", alpha=0.6, pad=1.6, lw=0))
    axes[r, 0].set_ylabel(Path(ds.images[pick[r]]).stem.split("_")[0], fontsize=6.4, labelpad=2)
fig.text(0.5, -0.01, "绿色 = 正确检出（TP）　红色 = 误检（FP）　蓝色 = 漏检（FN）",
         ha="center", fontsize=7.2, color="0.35")
fig.tight_layout()
fig.savefig(OUT / "qualitative.png", bbox_inches="tight")
plt.close(fig)

# ---------------- 2. 阈值敏感性 ----------------
m = {k: json.load(open(f"results/{k}/metrics.json")) for k in ("unet", "deepcrack")}
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), constrained_layout=True)
style = {"unet": ("U-Net", "#3B6EA5", "o", "-"), "deepcrack": ("DeepCrack", "#C0504D", "s", "--")}
for key, (label, color, mk, ls) in style.items():
    rows = m[key]["threshold_sweep"]
    t = [r["threshold"] for r in rows]
    ax[0].plot(t, [r["f1"] for r in rows], ls, marker=mk, ms=3, lw=1.1, color=color, label=label)
    ax[1].plot(t, [r["iou"] for r in rows], ls, marker=mk, ms=3, lw=1.1, color=color, label=label)
    best = m[key]["best_threshold"]
    for a in ax:
        a.axvline(best["threshold"], color=color, lw=0.7, ls=":", alpha=0.7)
for a, t in zip(ax, ["F1（像素级）", "IoU（像素级）"]):
    a.set_xlabel("分割阈值"); a.set_ylabel(t)
    a.grid(alpha=0.25, lw=0.4)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7.5,
           bbox_to_anchor=(0.5, 1.0), handlelength=1.8)
fig.savefig(OUT / "f1_threshold.png", bbox_inches="tight")
plt.close(fig)

# ---------------- 3. 训练曲线（含 loss） ----------------
hist = {"unet": ("U-Net", "#3B6EA5", "o", "-"), "deepcrack": ("DeepCrack", "#C0504D", "s", "--")}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3), constrained_layout=True)
for key, (label, color, mk, ls) in hist.items():
    rows = list(csv.DictReader(open(f"runs/{key}/history.csv")))
    ep = [int(r["epoch"]) for r in rows]
    ax[0].plot(ep, [float(r["train_loss"]) for r in rows], ls, marker=mk, ms=3, lw=1.1, color=color, label=label)
    ax[1].plot(ep, [float(r["val_f1"]) for r in rows], ls, marker=mk, ms=3, lw=1.1, color=color, label=label)
    ax[2].plot(ep, [float(r["val_iou"]) for r in rows], ls, marker=mk, ms=3, lw=1.1, color=color, label=label)
for a, t, use_log in zip(ax, ["训练损失", "验证 F1", "验证 IoU"], [True, False, False]):
    if use_log:
        a.set_yscale("log")
    a.set_xlabel("Epoch"); a.set_ylabel(t); a.grid(alpha=0.25, lw=0.4)
ax[0].legend(frameon=False, fontsize=7, loc="upper right")
fig.savefig(OUT / "training_curves.png", bbox_inches="tight")
plt.close(fig)

print("figures ok")