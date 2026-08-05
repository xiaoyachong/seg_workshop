# Segmentation workshop

Run a finetuned DINOv3 segmentation model on tomography slices and view the predictions.

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

Register the environment as a Jupyter kernel so the notebook can find it:

```bash
pip install ipykernel
python -m ipykernel install --user --name dino_demo --display-name "dino_demo"
```

## 2. Run inference

Open [jupyter.nersc.gov](https://jupyter.nersc.gov), browse to the `seg_workshop` folder you just
cloned, open `notebooks/inference.ipynb`, and pick the **dino_demo** kernel (top right).

Two cells:

1. **Settings** — set `DATA_DIR` to your dataset folder and `CKPT_PATH` to the checkpoint. Loads
   the model and prints the class/color table.
2. **Run** — predicts on the test split and plots image / prediction / ground truth for each
   image, inline. Nothing is saved to disk.

```python
DATA_DIR  = "../sample_dataset"
CKPT_PATH = "../out_sample_dataset/vits16-eomt-cityscapes/checkpoints/best.ckpt"
```

That's it — everything below is optional.

---

# For development

Only needed if you want to train your own model or prepare new data.

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

Scripts pick a split by falling back through a list, so either `val` or `test` is enough:

| script | train split | second split |
| --- | --- | --- |
| `finetune.py` | `train/` | `val/` → `valid/` → `test/` |
| `inference.py` | — | `test/` → `val/` → `valid/` |

## Train

```bash
python finetune.py sample_dataset
```

Reads `classes.json` for the class table, trains on `sample_dataset/train` with
`sample_dataset/val` for validation, passing `ignore_classes=[255]`. Checkpoints land in
`out_sample_dataset/vits16-eomt-cityscapes/`.

Edit `STEPS` at the top of `finetune.py`; `model` / `batch_size` / `devices` are arguments to the
`train_semantic_segmentation(...)` call. Uncomment `resume_interrupted=True` to continue a stopped
run — a fresh run into a non-empty output directory will fail.

`notebooks/finetune.ipynb` does the same thing interactively.

### On NERSC

```bash
mkdir -p logs                              # once — Slurm needs it at submit time
sbatch submit_reserved.sh sample_dataset   # inside the workshop reservation
sbatch submit.sh sample_dataset            # regular queue
```

`submit_reserved.sh` targets the workshop reservation and must use the **`amsc006_g`** account (the
GPU allocation — plain `amsc006` is refused). Add `--qos=shared` to fit four jobs per node instead
of one.

Monitor with `squeue --me`, then `tail -f logs/job_<jobid>.out`.

## Inference from the command line

```bash
python inference.py sample_dataset
```

Auto-discovers the newest `best.ckpt` under `out_sample_dataset/` (falling back to `last.ckpt`) and
writes one figure per image to `out_sample_dataset/inference/<name>_overlay.png`. Set `CKPT_PATH` at
the top of the script to pin a specific checkpoint.

## Inspect a dataset

`notebooks/view_image_mask_pairs.ipynb` — three cells: set `DATA_DIR`, plot every image/mask/overlay
pair, then print the distinct values in each mask with per-class pixel counts.

Use it to catch the common problems: masks that are mostly `ignore` (partially annotated slices),
class ids present in the pixels but not declared in `classes.json`, and duplicate labels sharing one
name across two ids.