#!/bin/bash
#SBATCH --job-name=hybrid-base
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=hybrid-base-%j.out

# Three small checks before any big hybrid run, all on the same mesh:
#   1. GPU only    (poisson_cube_bk3)           4 ranks, 1 per GPU
#   2. CPU only    (poisson_cube_cpu)           4 ranks, 1 per Grace
#   3. hybrid regression (switch_level=0)       4 ranks (k=1) and 16 ranks (k=4)
# Mesh: 3x3x3 coarse cells, 3 refinements = 24 cells/dir, Q4 -> 97^3 = 912673 DoFs.
# CG iterations and reductions of 1, 2 and 3 must agree (CPU may differ in the
# 3rd digit of the reduction: Chebyshev eigenvalue estimates).

REPO=$HOME/codes/dealiiX-portable-multigrid/tests
BK3_BIN=$REPO/poisson_cube_bk3/build/program
CPU_BIN=$REPO/poisson_cube_cpu/build/program
HYBRID_BIN=$REPO/poisson_cube_hybrid_gpu_cpu/build/program

module purge
module load Stages/2026
module load nvidia-compilers
module load OpenMPI
module load FlexiBLAS

# no NEON yet: plain dealii-mg-transfer, just checking that everything runs
export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-mg-transfer/lib:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

# exit-time abort (static Kokkos linked twice), skip the core files
ulimit -c 0

ldd $HYBRID_BIN | grep libdeal_II
nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"

# n ranks, k per Grace (= per GPU), 4k ranks on the node. Each rank gets
# 72/k cores; block placement then puts ranks 0..k-1 on the cores of Grace 0,
# k..2k-1 on Grace 1, ... (needs 72 % k == 0). No --ntasks-per-socket /
# --distribution: with them the step got cores of one Grace only, ranks were
# left unpinned and MPI latency was ~50 ms (job 2307102).
run() {
  local k=$1 bin=$2; shift 2
  local common="--ntasks=$((4 * k)) --cpus-per-task=$((72 / k)) --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
  echo
  echo "=============== k=$k: $bin $*"
  echo "placement:"
  srun $common bash -c 'echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V
  timeout 400 srun $common $bin "$@"
}

# poisson_cube_{bk3,cpu}: degree minsize maxsize n_pre n_post [s=non-doubling mesh] overlap
# min/max pick the one cycle with 24 cells/dir (sizes 20 and 28 give 531441 and 1442897)
MESH="900000 1000000 3 3 s 0"

echo "######## 1. GPU only"
run 1 $BK3_BIN 4 $MESH

echo "######## 2. CPU only"
run 1 $CPU_BIN 4 $MESH

echo "######## 3. hybrid, switch_level=0 (full GPU hierarchy)"
# args after k: degree n_refine switch_level n_subdivisions
run 1 $HYBRID_BIN 1 4 3 0 3
run 4 $HYBRID_BIN 4 4 3 0 3
