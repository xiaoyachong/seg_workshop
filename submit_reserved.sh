#!/bin/bash
#SBATCH -q regular
#SBATCH -A amsc006_g
#SBATCH --reservation=_CAP_als_annotation_workshop
#SBATCH -N 1
#SBATCH -C "gpu&hbm80g"
#SBATCH --time=10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Workshop run inside the reserved nodes.
#
#   mkdir -p logs          # once -- Slurm needs this to exist at submit time
#   sbatch submit_reserved.sh sample_dataset
#
# Reservation window: 2026-08-04 11:00 -> 2026-08-05 23:00, 10 nodes.
# The job is killed when the reservation ends, whatever --time says.
#
# The account MUST be amsc006_g (the GPU allocation) -- the reservation lists
# Accounts=amsc006_g, so plain "amsc006" is refused.
#
# To fit 4 attendees per node instead of 1, add "--qos=shared" at submit time;
# it is the one QOS that still has an effect inside a reservation.

set -euo pipefail

DATASET="${1:-sample_dataset}"
CONDA_ENV="${CONDA_ENV:-dino_workshop}"

export GPUS_PER_NODE=1
OUTPUT_DIR="out_$(basename "${DATASET}")"   # matches out= inside finetune.py

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
echo "Nodes:        ${SLURM_JOB_NODELIST:-local}"
echo "Job ID:       ${SLURM_JOB_ID:-none}"
echo "Reservation:  ${SLURM_JOB_RESERVATION:-none}"
echo "Account:      ${SLURM_JOB_ACCOUNT:-none}"
echo "GPUs:         $(( ${SLURM_NNODES:-1} * GPUS_PER_NODE ))"
echo "Dataset:      ${DATASET}"
echo "Output dir:   ${OUTPUT_DIR}"
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
