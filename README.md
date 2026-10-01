# Segmentation workshop

Finetune a DINOv3 segmentation model on tomography slices and view the predictions.

Everything you need is two notebooks: **`notebooks/finetune.ipynb`** to train, then
**`notebooks/inference.ipynb`** to look at the results.

---

## 1. Set up (once)

Log in to Perlmutter, then get the code:

```bash
git clone https://github.com/xiaoyachong/seg_workshop.git
cd seg_workshop
```

Create the environment and install the dependencies:

```bash
module load conda
conda create -n dino_demo python==3.12
conda activate dino_demo
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel so the notebooks can find it:

```bash
pip install ipykernel
python -m ipykernel install --user --name dino_demo --display-name "dino_demo"
```

## 2. Open the notebooks

Go to [jupyter.nersc.gov](https://jupyter.nersc.gov), browse to the `seg_workshop` folder you
cloned, and open a notebook from `notebooks/`. Pick the **dino_demo** kernel (top right) before
running anything.

Paths inside the notebooks are relative to `notebooks/`, so the repo root is `..` — a dataset at
`seg_workshop/data/combined` is written `"../data/combined"`.

## 3. Train — `notebooks/finetune.ipynb`

Edit the first cell, then run all cells:

```python
DATASET = "../data/combined"    # folder holding classes.json + train/ and val/
STEPS = 5000                    # training steps
MODEL = "dinov3/vits16-eomt-cityscapes"
```

Checkpoints are written to `notebooks/out_<dataset>/<model>/checkpoints/` — for the example above,
`notebooks/out_combined/vits16-eomt-cityscapes/checkpoints/best.ckpt`. The cell prints the exact
path. (Paths are relative to the notebook, so a notebook run writes inside `notebooks/`, whereas
`finetune.py` run from the repo root writes to `out_<dataset>/` there.)

Training in a notebook ties up the GPU for as long as it runs. For anything long, use the batch
script instead (see [On NERSC](#on-nersc) below) and come back to `inference.ipynb` afterwards.

## 4. Look at the results — `notebooks/inference.ipynb`

Set the same `DATA_DIR` and `MODEL` you trained with — the checkpoint path is worked out from
those two, so there's nothing else to fill in:

```python
DATA_DIR = "../data/combined"
MODEL = "dinov3/vits16-eomt-cityscapes"
OUT_ROOT = "."           # where finetune wrote out_<dataset>/
# CKPT_PATH is derived: ./out_combined/vits16-eomt-cityscapes/checkpoints/best.ckpt
```

It falls back to `last.ckpt` if `best.ckpt` isn't there yet, and prints the path it settled on.
Set `SPLIT` to choose which split to run on (`test`, `val`, …) and `ALPHA` for overlay opacity.

The second cell predicts on the test split and plots image / prediction / ground truth for each
image, inline. Nothing is saved to disk.

That's the whole workflow — everything below is optional.

---

# For development

## Other notebooks

| notebook | what it does |
| --- | --- |
| `merge_data.ipynb` | combine the workshop's per-person annotations into one dataset |
| `view_image_mask_pairs.ipynb` | inspect a dataset before training — plots pairs, prints mask label values |
| `inference_single_tiff.ipynb` | run a checkpoint on one image, no mask or dataset folder needed |
| `inference_folder.ipynb` | run over a folder of images, save masks and overlays |

`view_image_mask_pairs.ipynb` is worth a look before any training run — it catches masks that are
mostly `ignore` (partially annotated slices), class ids present in the pixels but not declared in
`classes.json`, and duplicate labels sharing one name across two ids.

## Dataset layout

```
sample_dataset/
├── classes.json        # {"0": "air", "1": "xylem", ...}  id -> label
├── train/
│   ├── images/         # xxx_0000.png, xxx_0001.png, ...
│   └── masks/          # same filenames as images/
├── val/                # optional; used for validation during training
│   ├── images/
│   └── masks/
└── test/               # optional; what inference runs on
    ├── images/
    └── masks/          # optional — enables the ground-truth panel
```

An image and its mask are paired by **identical filename**. Masks are single-channel `uint8` PNGs
where the **pixel value is the class id** from `classes.json`.

**`255` means unannotated** — never painted, excluded from the loss, and deliberately absent from
`classes.json`. Every script hardcodes this.

**`classes.json` is the only metadata read.** Nothing reads `manifest.json`; its `classId` values
can be offset from the ids in the mask pixels, so trusting it would mislabel classes. Overlay
colors are assigned from a `tab20` palette by class index instead.

Splits are picked by falling back through a list, so either `val` or `test` is enough:

| script | train split | second split |
| --- | --- | --- |
| `finetune.py` | `train/` | `val/` → `valid/` → `test/` |
| `inference.py` | — | `test/` → `val/` → `valid/` |

## Command line

```bash
python finetune.py sample_dataset     # train; checkpoints -> out_sample_dataset/
python inference.py sample_dataset    # overlays -> out_sample_dataset/inference/
```

Same behaviour as the notebooks. `finetune.py` takes `STEPS` from the top of the file; `model`,
`batch_size` and `devices` are arguments to the `train_semantic_segmentation(...)` call. Uncomment
`resume_interrupted=True` to continue a stopped run — a fresh run into a non-empty output directory
will fail.

Note that `inference.py` derives its output folder from the argument as given, while `finetune.py`
uses only the last path component. Passing a nested path like `data/earth` sends training to
`out_earth/` but makes `inference.py` look in `out_data/earth/`. Use the notebook, or pass a
top-level folder name.

### On NERSC

```bash
mkdir -p logs                              # once — Slurm needs it at submit time
sbatch submit_reserved.sh sample_dataset   # inside the workshop reservation
sbatch submit.sh sample_dataset            # regular queue
```

`submit_reserved.sh` targets the workshop reservation and must use the **`amsc006_g`** account (the
GPU allocation — plain `amsc006` is refused). Add `--qos=shared` to fit four jobs per node instead
of one. Pick a different conda environment with `CONDA_ENV=/full/path/to/env sbatch ...`.

Monitor with `squeue --me`, then `tail -f logs/job_<jobid>.out`.
