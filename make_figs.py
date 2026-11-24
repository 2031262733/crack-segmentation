import csv, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 300, "svg.fonttype": "none"})
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), constrained_layout=True)
runs = {"U-Net": ("runs/unet/history.csv", "#3B6EA5", "o", "-"),
        "DeepCrack": ("runs/deepcrack/history.csv", "#C0504D", "s", "--")}
for name, (p, c, m, ls) in runs.items():
    rows = list(csv.DictReader(open(p)))
    ep = [int(r["epoch"]) for r in rows]
    ax[0].plot(ep, [float(r["val_f1"]) for r in rows], ls, marker=m, ms=3, lw=1.1, color=c, label=name)
    ax[1].plot(ep, [float(r["val_iou"]) for r in rows], ls, marker=m, ms=3, lw=1.1, color=c, label=name)
for a, t in zip(ax, ["Validation F1", "Validation IoU"]):
    a.set_xlabel("Epoch"); a.set_ylabel(t); a.legend(frameon=False, fontsize=7, loc="lower right")
    a.grid(alpha=0.25, lw=0.4)
fig.savefig("docs/curves.png", bbox_inches="tight")
fig.savefig("docs/curves.svg", bbox_inches="tight")
print("ok")

# 汇总表
results = {"unet": "results/unet/metrics.json", "deepcrack": "results/deepcrack/metrics.json"}
for n, p in results.items():
    r = json.load(open(p))
    a = r["at_0.5"]
    print(f'{n}: P {a["precision"]:.3f} R {a["recall"]:.3f} F1 {a["f1"]:.3f} IoU {a["iou"]:.3f} '
          f'imgF1 {a["f1_image_mean"]:.3f} FPS {r["fps_batch1"]:.0f} params {r["params_m"]}M')
