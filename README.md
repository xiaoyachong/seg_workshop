# Segmentation finetuning workshop

Finetune a DINOv3 semantic segmentation model on an annotated tomography dataset, then run
inference and save overlays. Everything is driven by a single dataset folder name.

## Install

```bash
conda create -n dino_demo python==3.12
conda activate dino_demo
pip install -r requirements.txt
```

## Dataset layout

Point the scripts at an exported dataset folder — `sample_dataset` in the examples below:

```
sample_dataset/
├── classes.json        # {"0": "air", "1": "xylem", ...}  id -> label
├── train/
│   ├── images/         # xxx_0000.png, xxx_0001.png, ...
│   └── masks/          # same filenames as images/
├── val/                # optional; used for validation during training
│   ├── images/
│   └── masks/
└── test/               # optional; what inference.py runs on
    ├── images/
    └── masks/          # optional — enables the ground-truth panel
```

An image and its mask are paired by **identical filename**. Masks are single-channel `uint8`
PNGs where the **pixel value is the class id** from `classes.json`.

Both scripts pick a split by falling back through a list, so either `val` or `test` is enough:

| script | train split | second split |
| --- | --- | --- |
| `finetune.py` | `train/` | `val/` → `valid/` → `test/` |
| `inference.py` | — | `test/` → `val/` → `valid/` |

### Class ids and unannotated pixels

**`255` means unannotated.** Those pixels were never painted, are excluded from the loss, and are
deliberately absent from `classes.json`. Every script hardcodes this, so no manifest is required.

**`classes.json` is the only metadata read.** Nothing reads `manifest.json` — it may sit in the
folder, but its `classId` values can be offset from the ids in the mask pixels, so trusting it
would mislabel classes. Overlay colors are assigned from a `tab20` palette by class index instead,
which also keeps repeated labels distinguishable.

## Merge several datasets

```bash
python merge_datasets.py rock earth plant -o combined
```

Combines same-structure folders into one dataset you can train on directly. Files are copied as
`<folder>_<filename>` so names can't collide, and every class gets a new global id labelled
`<folder>_<classname>` — `rock_background`, `plant_xylem`, and so on. Labels are copied verbatim
apart from the prefix, so a folder that declares the same label twice keeps both; they simply get
separate global ids. Classes are never fused across datasets, even when they share a name.

Mask pixels are remapped through a lookup table into the global id space. `255` is the ignore value
in every input and stays `255` in the output. Values found in a mask but declared nowhere are sent
to ignore with a warning.

The output carries its own `classes.json`, so `finetune.py`, `inference.py`, and the notebook all
read it like any single export:

```bash
python merge_datasets.py rock earth plant -o combined
python finetune.py combined
```

## Train

```bash
python finetune.py sample_dataset
```

Reads `classes.json` for the class table, then trains on `sample_dataset/train` with
`sample_dataset/val` for validation, passing `ignore_classes=[255]`. Checkpoints land in
`out_sample_dataset/vits16-eomt-cityscapes/`.

Edit at the top of `finetune.py`: `STEPS`, and the `model` / `batch_size` / `devices` arguments.
Uncomment `resume_interrupted=True` to continue a stopped run.

### On a SLURM cluster

```bash
sbatch submit.sh sample_dataset
```

Single node, single GPU. The dataset name is the only argument; `submit.sh` checks that
`sample_dataset/classes.json` exists before burning an allocation, creates `logs/`, and writes
job output to `logs/job_<jobid>.out`.

Set the conda environment with `CONDA_ENV=/path/to/env sbatch submit.sh sample_dataset`, and edit
the `#SBATCH` header for account, walltime, and queue.

## Inference

```bash
python inference.py sample_dataset
```

Auto-discovers the newest `best.ckpt` under `out_sample_dataset/` (falling back to `last.ckpt`), runs the
model over the test split, and writes one figure per image — image / prediction / ground truth —
to `out_sample_dataset/inference/<name>_overlay.png`.

Set `CKPT_PATH` at the top of the script to use a specific checkpoint instead, or `ALPHA` to change
the overlay opacity.

### In a notebook

`inference.ipynb` does the same thing interactively. Set `DATA_DIR` and `CKPT_PATH` in the first cell (no auto-discovery), then run the remaining.

To use it on NERSC, register the conda environment as a Jupyter kernel once:

```bash
conda activate dino_demo
pip install ipykernel
python -m ipykernel install --user --name dino_demo --display-name "dino_demo"
```

Then open the notebook at [jupyter.nersc.gov](https://jupyter.nersc.gov) and pick the **dino_demo**
kernel.

## Inspect a dataset first

`view_image_mask_pairs.ipynb` is a three-cell notebook for sanity-checking an export before
training:

1. Set `DATA_DIR` and load the class table
2. Plot every image / mask / overlay pair
3. Print the distinct values in each mask, with per-class pixel counts

Use it to catch the common problems: masks that are mostly `ignore` (partially annotated slices),
class ids that appear in the pixels but aren't declared in `classes.json`, and duplicate labels
sharing one name across two ids.

## Files

| file | purpose |
| --- | --- |
| `merge_datasets.py` | combine several dataset folders into one global label space |
| `finetune.py` | train a segmentation model on a dataset folder |
| `submit.sh` | SLURM wrapper around `finetune.py` (1 node, 1 GPU) |
| `inference.py` | run a trained checkpoint over the test split, save overlays |
| `inference.ipynb` | same, interactively — plots inline, nothing saved |
| `view_image_mask_pairs.ipynb` | visualize pairs and inspect mask label values |
| `requirements.txt` | pinned dependencies |
