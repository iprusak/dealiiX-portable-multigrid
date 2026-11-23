#!/bin/bash
#SBATCH --job-name=pop-coe
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=pop-coe-%j.out
#
# Nsight Systems profile of the BK3 multigrid solve: Q4, doubling mesh, 5 + 5
# smoothing, float V-cycle, 10 CG solves on one mesh, 1 rank per GPU. The program
# takes the largest size that fits, 270M DoFs per GPU: 1076890625 DoFs on the 4
# GPUs of a node. One report per rank in pop-coe-<job>/.
#
# Build and submit from this directory:
#
#   source ../../scripts/jupiter/env.sh [DEAL_II_DIR]
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=$DEAL_II_DIR
#   cmake --build build
#   sbatch run_sbatch_jupiter.sh
#
# Several nodes: sbatch --nodes=<N> run_sbatch_jupiter.sh. Another build
# directory: BUILD_DIR=<dir> sbatch run_sbatch_jupiter.sh.

set -u

BUILD_DIR=${BUILD_DIR:-build}
NODES=${SLURM_JOB_NUM_NODES:-1}
OUT=$PWD/pop-coe-$SLURM_JOB_ID

# module stack, LD_LIBRARY_PATH and thread pinning (DEAL_II_DIR from the
# environment or its default in env.sh)
source ../../scripts/jupiter/env.sh ${DEAL_II_DIR:-}
module load Nsight-Systems/2025.5.1

export NVCOMPILER_OMP_DISABLE_WARNINGS=true

mkdir -p $OUT
ldd $BUILD_DIR/program | grep -E 'libdeal_II|libkokkoscore'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srun --mpi=pspmix --nodes=$NODES --ntasks=$((4 * NODES)) --ntasks-per-node=4 \
  --cpus-per-task=72 --distribution=block:block --cpu-bind=cores \
  --gpu-bind=closest --mem-bind=local \
  nsys profile -o $OUT/NSYS_%q{SLURM_PROCID} -t cuda,nvtx,mpi \
  $BUILD_DIR/program
