import os

os.environ["LIGHTLY_TRAIN_CACHE_DIR"] = ".cache"
os.environ["LIGHTLY_TRAIN_MODEL_CACHE_DIR"] = ".cache"
os.environ["TORCH_HOME"] = ".cache"

import json
import sys
from pathlib import Path

import lightly_train

# Dataset folder: holds classes.json, manifest.json and train/ + val/ (or test/).
# Override from the command line:  python finetune.py sample_dataset
DATASET = sys.argv[1] if len(sys.argv) > 1 else "sample_dataset"

ROOT = Path(DATASET)
STEPS = 10000


def load_classes(root: Path):
    """Read the class table and ignore value straight from the export.

    classes.json maps id -> label, and those ids are what the mask pixels contain.
    manifest.json may declare an ignore_index (255 in newer exports) marking pixels
    the annotator never painted; older exports have no such value.
    """
    classes = {int(k): v for k, v in json.loads((root / "classes.json").read_text()).items()}

    ignore = None
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        ignore = json.loads(manifest_path.read_text()).get("ignore_index")

    if ignore is not None:
        classes[int(ignore)] = classes.get(int(ignore), "unknown")

    return classes, ([int(ignore)] if ignore is not None else [])


def split_dir(root: Path, names) -> Path:
    for name in names:
        if (root / name / "images").is_dir():
            return root / name
    raise FileNotFoundError(f"none of {names} found under {root}")


CLASSES, IGNORE_CLASSES = load_classes(ROOT)
TRAIN = split_dir(ROOT, ["train"])
VAL = split_dir(ROOT, ["val", "valid", "test"])

if __name__ == "__main__":
    print(f"dataset       : {ROOT}")
    print(f"train / val   : {TRAIN.name} / {VAL.name}")
    print(f"classes       : {CLASSES}")
    print(f"ignore_classes: {IGNORE_CLASSES}")

    lightly_train.train_semantic_segmentation(
        out=f"out_{ROOT.name}/vits16-eomt-cityscapes",
        model="dinov3/vits16-eomt-cityscapes",
        steps=STEPS,
        devices=1,
        num_nodes=1,
        batch_size=1,
        data={
            "train": {
                "images": str(TRAIN / "images"),
                "masks": str(TRAIN / "masks"),
            },
            "val": {
                "images": str(VAL / "images"),
                "masks": str(VAL / "masks"),
            },
            "classes": CLASSES,
            "ignore_classes": IGNORE_CLASSES,
        },
        logger_args={
            "log_every_num_steps": 100,
            "val_every_num_steps": 100,
            "val_log_every_num_steps": 100,
        },
        save_checkpoint_args={
            "save_every_num_steps": 1000,
            "save_last": True,
            "save_best": True,
        },
        # resume_interrupted=True,
    )
