"""Evaluate a trained checkpoint on the official CFD test split.

    python evaluate.py --ckpt runs/deepcrack/best.pt --out results/deepcrack

Writes metrics.json (threshold sweep, per-image and micro metrics, params, FPS)
and four-panel overlay figures under results/<model>/overlays/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import CrackDataset
from losses import confusion, scores
from models import build_model


@torch.no_grad()
def collect(model, loader, device):
    """Return per-image probability maps and targets (kept in RAM: 1695 x 1 x 256 x 256)."""
    probs, gts = [], []
    for x, y in loader:
        p = torch.sigmoid(model(x.to(device))[-1]).cpu()
        probs.append(p)
        gts.append(y)
    return torch.cat(probs), torch.cat(gts)


def sweep(probs, gts, thrs=np.arange(0.1, 0.91, 0.05)):
    rows = []
    for t in thrs:
        tp = fp = fn = 0.0
        f1s = []
        for p, g in zip(probs, gts):
            pred = (p > t).float()
            a = float((pred * g).sum())
            b = float((pred * (1 - g)).sum())
            c = float(((1 - pred) * g).sum())
            tp, fp, fn = tp + a, fp + b, fn + c
            m = scores(a, b, c)
            f1s.append(m["f1"])
        m = scores(tp, fp, fn)
        m.update(threshold=float(t), f1_image_mean=float(np.mean(f1s)))
        rows.append(m)
    return rows


@torch.no_grad()
def measure_fps(model, device, size=256, runs=200):
    model.eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(20):
        model(x)
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    torch.cuda.synchronize() if device == "cuda" else None
    return runs / (time.perf_counter() - t0)


def save_overlays(probs, gts, ds, out_dir, n=6, thr=0.5):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    for k, i in enumerate(idx):
        img = np.array(Image.open(ds.images[i]).convert("RGB").resize((probs.shape[-1],) * 2))
        g = gts[i, 0].numpy() > 0.5
        p = probs[i, 0].numpy() > thr
        gt = (g * 255).astype(np.uint8)
        pr = (p * 255).astype(np.uint8)
        # TP 绿 / FP 红 / FN 蓝，叠在原图上
        err = img.copy()
        err[g & p] = (0.35 * err[g & p] + 0.65 * np.array([0, 220, 0])).astype(np.uint8)
        err[p & ~g] = (0.35 * err[p & ~g] + 0.65 * np.array([230, 40, 40])).astype(np.uint8)
        err[g & ~p] = (0.35 * err[g & ~p] + 0.65 * np.array([40, 90, 230])).astype(np.uint8)
        canvas = np.concatenate([img, np.stack([gt] * 3, -1), np.stack([pr] * 3, -1), err], 1)
        Image.fromarray(canvas).save(out_dir / f"pred_{k:02d}_{ds.images[i].stem}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/cfd")
    ap.add_argument("--size", type=int, default=0, help="0 = use the size stored in the checkpoint")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0, help="Windows spawn 下每个 worker 约 1 GB 提交内存，内存紧张时保持 0")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    size = args.size or ck.get("size", 256)
    model = build_model(ck["model"], pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    ds = CrackDataset(args.data, "test", size, train=False)
    ld = DataLoader(ds, args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    probs, gts = collect(model, ld, device)

    rows = sweep(probs, gts)
    best = max(rows, key=lambda r: r["f1"])
    at50 = next(r for r in rows if abs(r["threshold"] - 0.5) < 1e-6)
    fps = measure_fps(model, device, size)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "model": ck["model"],
        "checkpoint_epoch": ck.get("epoch"),
        "val_f1": ck.get("val_f1"),
        "size": size,
        "test_images": len(ds),
        "at_0.5": {k: round(v, 4) for k, v in at50.items()},
        "best_threshold": {k: round(v, 4) for k, v in best.items()},
        "fps_batch1": round(fps, 1),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "threshold_sweep": [{k: round(v, 4) for k, v in r.items()} for r in rows],
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    save_overlays(probs, gts, ds, out / "overlays")


if __name__ == "__main__":
    main()
