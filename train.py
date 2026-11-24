"""Train a crack segmentation model on CFD.

Examples:
    python train.py --model unet      --epochs 12
    python train.py --model deepcrack --epochs 10 --amp 0 --freeze-bn 1 --side-weight 0.25 --clip 1.0

Outputs: runs/<model>/best.pt, history.csv, config.json
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import CrackDataset, split_indices
from losses import confusion, deep_supervision_loss, scores
from models import build_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    tp = fp = fn = 0.0
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += deep_supervision_loss(out, y).item()
        n += 1
        a, b, c, _ = confusion(out[-1], y, thr)
        tp, fp, fn = tp + a, fp + b, fn + c
    m = scores(tp, fp, fn)
    m["loss"] = loss_sum / max(n, 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet", choices=["unet", "deepcrack"])
    ap.add_argument("--data", default="data/cfd")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    # Windows spawn 下每个 worker 都会重新 import torch（约 1 GB 提交内存）；
    # 16 GB 内存 + 小页面文件时会直接崩在 WinError 1455（页面文件太小）。默认 0，内存充裕再调大。
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    # 预训练 VGG16 的 BN 统计量来自 ImageNet；batch 4 时若继续更新 running stats，
    # 几百步内就会被小批量统计带偏，随后训练发散。迁移微调时冻结 BN 更稳。
    ap.add_argument("--freeze-bn", type=int, default=0)
    # DeepCrack 有 6 个输出，各项加权求和后梯度尺度是 U-Net 的 2–3 倍；fp16 下 Dice 的
    # 全局求和与 6 项叠加都容易溢出成 inf，实测第 4 轮开始发散、第 7 轮变 NaN。
    # 因此把损失放到 fp32 算，并提供 --amp 0 直接关混合精度。
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--side-weight", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=5.0)
    args = ap.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    train_full = CrackDataset(args.data, "train", args.size, train=True)
    val_full = CrackDataset(args.data, "train", args.size, train=False)
    tr_idx, va_idx = split_indices(len(train_full))
    train_ds = Subset(train_full, tr_idx)
    val_ds = Subset(val_full, va_idx)
    print(f"train {len(train_ds)} / val {len(val_ds)} / size {args.size}")

    train_ld = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)

    torch.cuda.empty_cache()
    model = build_model(args.model).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{args.model}: {n_param:.1f} M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    use_amp = bool(args.amp) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist = []
    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
        t0, run_loss, nb, gn_max = time.time(), 0.0, 0, 0.0
        pbar = tqdm(train_ld, desc=f"ep {ep}/{args.epochs}", ncols=90)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
            # 损失永远在 fp32 里算：fp16 的 Dice 全局求和会溢出
            loss = deep_supervision_loss([o.float() for o in logits], y, side_weight=args.side_weight)
            if not torch.isfinite(loss):
                # 单步出现 inf/NaN 时直接跳过这一步，避免污染权重（日志里会留下痕迹）
                print(f"  skip non-finite loss at iter {nb}")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            gn_max = max(gn_max, float(gn))
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            nb += 1
            pbar.set_postfix(loss=f"{run_loss / nb:.4f}")
        sched.step()

        val = evaluate(model, val_ld, device)
        row = {"epoch": ep, "train_loss": run_loss / max(nb, 1), "lr": opt.param_groups[0]["lr"],
               "grad_norm_max": round(gn_max, 2),
               "time_s": time.time() - t0, **{f"val_{k}": v for k, v in val.items()}}
        hist.append(row)
        print(f"ep {ep}: train {row['train_loss']:.4f} | val F1 {val['f1']:.4f} IoU {val['iou']:.4f} "
              f"({row['time_s']:.0f}s)")

        if val["f1"] > best:
            best = val["f1"]
            torch.save({"model": args.model, "state_dict": model.state_dict(),
                        "size": args.size, "epoch": ep, "val_f1": best}, out_dir / "best.pt")
            print(f"  saved best (F1 {best:.4f})")

        with open(out_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writeheader()
            w.writerows(hist)

    print(f"done. best val F1 = {best:.4f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
