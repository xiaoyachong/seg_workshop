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
├── manifest.json       # class colors, annotator, and (newer exports) "ignore_index"
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

Two details differ between exports, and both scripts read them rather than assume:

- **Colors join on the label, not the id.** `classes.json` ids are what the mask pixels contain,
  but `manifest.json`'s `classId` can be offset from them (in one export `classes.json` starts at
  `0: air` while `manifest.json` starts at `1: air`). Joining on the number silently mislabels
  every class.
- **`ignore_index` marks unannotated pixels.** Newer exports declare `"ignore_index": 255` in
  `manifest.json`; those pixels were never painted and are excluded from the loss. Older exports
  have no such value, so unpainted pixels fall through to id `0` and cannot be distinguished from
  real background — worth checking before you train.

## Train

```bash
python finetune.py sample_dataset
```

Reads `classes.json` for the class table and `manifest.json` for `ignore_index`, then trains on
`sample_dataset/train` with `sample_dataset/val` for validation. Checkpoints land in
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
| `finetune.py` | train a segmentation model on a dataset folder |
| `submit.sh` | SLURM wrapper around `finetune.py` (1 node, 1 GPU) |
| `inference.py` | run a trained checkpoint over the test split, save overlays |
| `view_image_mask_pairs.ipynb` | visualize pairs and inspect mask label values |
| `requirements.txt` | pinned dependencies |
