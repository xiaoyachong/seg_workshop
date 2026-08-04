"""Run a finetuned model over a dataset's test split and save overlays.

    python inference.py sample_dataset
"""

import os

os.environ["LIGHTLY_TRAIN_CACHE_DIR"] = ".cache"
os.environ["LIGHTLY_TRAIN_MODEL_CACHE_DIR"] = ".cache"
os.environ["TORCH_HOME"] = ".cache"

import colorsys
import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import lightly_train

DATASET = sys.argv[1] if len(sys.argv) > 1 else "sample_dataset"

ROOT = Path(DATASET)
OUT_DIR = Path(f"out_{DATASET}")          # where finetune.py wrote its checkpoints
SAVE_DIR = OUT_DIR / "inference"          # overlays are written here
ALPHA = 0.45
CKPT_PATH = None                          # set to a path to skip auto-discovery

Image.MAX_IMAGE_PIXELS = None
IGNORE = 255              # unannotated pixels; drawn transparent
IGNORE_COLOR = (255, 0, 255)


def auto_color(n: int):
    """A distinct color per class index, for any number of classes."""
    if n < 20:
        return [int(c * 255) for c in plt.get_cmap("tab20")(n)[:3]]
    r, g, b = colorsys.hsv_to_rgb((n * 0.618033988749895) % 1.0, 0.65, 0.95)
    return [int(r * 255), int(g * 255), int(b * 255)]


# --- dataset ---------------------------------------------------------------
# classes.json maps id -> label, and those ids are what the mask pixels hold.
# Colors are assigned here rather than read from the export, so labels that
# repeat (0 and 1 both "background") still get distinguishable shades.
CLASSES = {int(k): v for k, v in json.loads((ROOT / "classes.json").read_text()).items()}

# Built once and indexed by class id, so a class keeps the same color in every
# figure -- never derive colors from whatever happens to appear in one image.
PALETTE = np.zeros((256, 3), dtype=np.uint8)
for n, i in enumerate(sorted(CLASSES)):
    PALETTE[i] = auto_color(n)
PALETTE[IGNORE] = IGNORE_COLOR

name = lambda i: "ignore/unannotated" if i == IGNORE else CLASSES.get(int(i), f"UNDECLARED {i}")

EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}
is_image = lambda p: p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith(".")


def split_dir(root: Path, names) -> Path:
    for n in names:
        if (root / n / "images").is_dir():
            return root / n
    raise FileNotFoundError(f"none of {names} found under {root}")


def find_ckpt() -> str:
    if CKPT_PATH:
        return CKPT_PATH
    for pattern in ("**/checkpoints/best.ckpt", "**/checkpoints/last.ckpt"):
        found = sorted(OUT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if found:
            return str(found[0])
    raise FileNotFoundError(f"no checkpoint under {OUT_DIR} -- set CKPT_PATH")


def normalize(img, lo=2, hi=98):
    """Percentile stretch so faint tomography slices are actually visible."""
    a, b = np.percentile(img, lo), np.percentile(img, hi)
    if b <= a:
        return img.astype(np.uint8)
    return (np.clip(img, a, b) - a) / (b - a) * 255


if __name__ == "__main__":
    TEST = split_dir(ROOT, ["test", "val", "valid"])
    images = sorted(p for p in (TEST / "images").iterdir() if is_image(p))
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = find_ckpt()
    print(f"dataset    : {ROOT}  ({TEST.name} split, {len(images)} images)")
    print(f"checkpoint : {ckpt}")
    print(f"saving to  : {SAVE_DIR}")
    print("colors     : fixed for every image")
    for i in sorted(CLASSES) + [IGNORE]:
        print("               {:>3}  {:<28} #{:02x}{:02x}{:02x}".format(i, name(i), *PALETTE[i]))

    model = lightly_train.load_model_from_checkpoint(ckpt)

    for img_path in images:
        image = normalize(np.array(Image.open(img_path).convert("RGB"))).astype(np.uint8)

        pred = model.predict(str(img_path))
        pred = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
        pred = pred.astype(np.int64)

        mask_path = TEST / "masks" / img_path.name
        gt = np.array(Image.open(mask_path).convert("L")) if is_image(mask_path) else None

        panels = [("image", image, None), ("prediction", image, pred)]
        if gt is not None:
            panels.append(("ground truth", image, gt))

        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5.5))
        for ax, (title, base, overlay) in zip(np.atleast_1d(axes), panels):
            ax.imshow(base)
            if overlay is not None:
                # RGBA so ignore pixels can be fully transparent, everything else ALPHA
                rgba = np.dstack([PALETTE[overlay], np.full(overlay.shape, int(255 * ALPHA), np.uint8)])
                rgba[..., 3][overlay == IGNORE] = 0
                ax.imshow(rgba, interpolation="nearest")
            ax.set_title(title)
            ax.axis("off")

        ids = sorted(set(np.unique(pred).tolist()) | (set(np.unique(gt).tolist()) if gt is not None else set()))
        np.atleast_1d(axes)[-1].legend(
            handles=[mpatches.Patch(color=PALETTE[i] / 255, label=f"{i}  {name(i)}") for i in ids],
            loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8,
        )

        fig.suptitle(img_path.name, fontsize=11)
        fig.tight_layout()
        out_path = SAVE_DIR / f"{img_path.stem}_overlay.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out_path}")

    print(f"\ndone -- {len(images)} overlays in {SAVE_DIR}")
