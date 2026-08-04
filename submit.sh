#!/bin/bash
#SBATCH -q regular
#SBATCH -A als_g
#SBATCH -N 1                          # 1 node = 1 GPU (single-GPU run)
#SBATCH -C gpu
#SBATCH --time=20:00:00               # adjust after a test run
#SBATCH --gres=gpu:1                  # 1 GPU on this node
#SBATCH --ntasks-per-node=1           # must match number of GPUs (LightlyTrain's SLURM recipe)
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/job_%j.out      # %j = job ID
#SBATCH --error=logs/job_%j.err

# Single-node, single-GPU finetuning run.
#
#   sbatch submit.sh sample_dataset
#
# The dataset folder name is the one argument -- it must contain classes.json
# plus train/ and val/ (or test/). finetune.py derives everything else from it.
#
# NOTE: on Perlmutter, "-q regular -C gpu" without "-q shared" typically
# allocates (and bills for) the whole 4-GPU node regardless of --gres=gpu:1.
# If your allocation supports "-q shared" for GPU nodes, switch to that so
# you're only billed for the 1 GPU you actually use.

set -euo pipefail

DATASET="${1:-sample_dataset}"
CONDA_ENV="${CONDA_ENV:-/pscratch/sd/x/xchong/envs/dino_demo}"

export GPUS_PER_NODE=1
OUTPUT_DIR="out_${DATASET}"           # must match the out= path inside finetune.py

if [ ! -f "${DATASET}/classes.json" ]; then
    echo "ERROR: ${DATASET}/classes.json not found -- is '${DATASET}' the right folder?" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" logs

module load conda
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "JOB STARTED: $(date)"
echo "============================================================"
echo "Nodes:      ${SLURM_JOB_NODELIST:-local}"
echo "Job ID:     ${SLURM_JOB_ID:-none}"
echo "GPUs:       $(( ${SLURM_NNODES:-1} * GPUS_PER_NODE ))"
echo "Dataset:    ${DATASET}"
echo "Output dir: ${OUTPUT_DIR}"
echo "============================================================"

START_TIME=$(date +%s)

srun python finetune.py "${DATASET}"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "JOB COMPLETED: $(date)"
echo "============================================================"
echo "Total time: $((DURATION / 60))m $((DURATION % 60))s (${DURATION}s)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
