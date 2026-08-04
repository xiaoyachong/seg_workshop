"""Merge several exported datasets into one combined dataset.

    python merge_datasets.py rock earth plant -o combined

Each input folder must look like:

    rock/
    ├── classes.json      # id -> label (255 = ignore, not listed)
    ├── train/{images,masks}/
    ├── val/{images,masks}/     (optional)
    └── test/{images,masks}/    (optional)

The merged output has the same shape. Files are renamed `<folder>_<filename>` so
names from different datasets can't collide, and every class is given a new global
id with the label `<folder>_<classname>`, copied verbatim apart from the prefix.
Mask pixels are remapped through a lookup table into that global id space; 255 is
the ignore value in every input and stays 255 in the output.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

IGNORE = 255  # global ignore id in the merged dataset
SPLITS = ["train", "val", "test"]
EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith(".")


def read_mask(path: Path) -> np.ndarray:
    a = np.asarray(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def load_classes(root: Path) -> dict:
    """id -> label. These ids are what the mask pixels contain.

    255 is the ignore value in every input and is not listed in classes.json.
    """
    return {int(k): v for k, v in json.loads((root / "classes.json").read_text()).items()}


def build_global_classes(datasets):
    """Assign every input class a new global id, labelled <folder>_<classname>.

    Returns (global_classes, luts) where luts[name] is a 256-entry array mapping
    that dataset's local mask values to global ids.
    """
    global_classes, luts = {}, {}
    next_id = 0

    for root in datasets:
        classes = load_classes(root)
        lut = np.full(256, IGNORE, dtype=np.uint8)  # anything unmapped -> ignore

        for local_id in sorted(classes):
            if local_id == IGNORE:
                continue  # the ignore value folds straight through

            if next_id >= IGNORE:
                sys.exit(f"ERROR: more than {IGNORE} classes total -- global ids would collide with ignore")

            # Labels are copied as-is with a folder prefix; duplicates within a
            # dataset stay duplicates, they just get separate global ids.
            global_classes[next_id] = f"{root.name}_{classes[local_id]}"
            lut[local_id] = next_id
            next_id += 1

        luts[root.name] = lut

    return global_classes, luts


def merge_split(root: Path, split: str, lut: np.ndarray, out: Path, declared: set) -> int:
    """Copy one split's images and write remapped masks. Returns the pair count."""
    img_dir, mask_dir = root / split / "images", root / split / "masks"
    if not img_dir.is_dir():
        return 0

    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "masks").mkdir(parents=True, exist_ok=True)

    count = 0
    for img in sorted(p for p in img_dir.iterdir() if is_image(p)):
        mask_path = mask_dir / img.name
        if not is_image(mask_path):
            print(f"    WARNING: no mask for {img.name}, skipping")
            continue

        mask = read_mask(mask_path)
        unknown = sorted(set(np.unique(mask).tolist()) - declared)
        if unknown:
            print(f"    WARNING: {img.name} has undeclared values {unknown} -> ignore")

        new_name = f"{root.name}_{img.name}"
        shutil.copy2(img, out / split / "images" / new_name)
        Image.fromarray(lut[mask]).save(out / split / "masks" / new_name)
        count += 1

    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("datasets", nargs="+", help="dataset folders to merge")
    ap.add_argument("-o", "--out", default="combined", help="output folder (default: combined)")
    args = ap.parse_args()

    roots = [Path(d.rstrip("/")) for d in args.datasets]
    for r in roots:
        if not (r / "classes.json").is_file():
            sys.exit(f"ERROR: {r}/classes.json not found")

    out = Path(args.out)
    global_classes, luts = build_global_classes(roots)

    print(f"merging {len(roots)} datasets into {out}/\n")
    print("global classes:")
    for gid, label in global_classes.items():
        print(f"  {gid:>3}: {label}")
    print(f"  {IGNORE}: ignore/unannotated\n")

    totals = {s: 0 for s in SPLITS}
    for root in roots:
        classes = load_classes(root)
        declared = set(classes) | {IGNORE}
        print(f"{root.name}  (local ids {sorted(classes)}, ignore={IGNORE})")
        for split in SPLITS:
            n = merge_split(root, split, luts[root.name], out, declared)
            totals[split] += n
            if n:
                print(f"    {split:<6} {n} pairs")
        print()

    # Same convention as the inputs: 255 is the ignore value and is deliberately
    # absent from classes.json.
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps(
        {str(gid): label for gid, label in global_classes.items()}, indent=2))

    print("totals: " + ", ".join(f"{s}={totals[s]}" for s in SPLITS if totals[s]))
    print(f"wrote {out}/classes.json ({len(global_classes)} classes, ignore={IGNORE})")
    print(f"\nnow run:  python finetune.py {out}")


if __name__ == "__main__":
    main()
