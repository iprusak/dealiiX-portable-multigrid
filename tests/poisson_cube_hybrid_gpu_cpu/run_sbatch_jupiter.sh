#!/bin/bash
#SBATCH --job-name=hybrid-sweep
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=hybrid-%j.out

module purge
module load Stages/2026
module load nvidia-compilers
module load OpenMPI
module load FlexiBLAS

# no NEON yet: plain dealii-mg-transfer, just checking that the hybrid runs
export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-mg-transfer/lib:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

# exit-time abort (static Kokkos linked twice), skip the core files
ulimit -c 0

ldd ./program | grep libdeal_II
nvidia-smi --query-gpu=index,compute_mode --format=csv,noheader
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"

# n ranks, k per Grace (= per GPU), 4k ranks on the node. Each rank gets
# 72/k cores; block placement then puts ranks 0..k-1 on the cores of Grace 0,
# k..2k-1 on Grace 1, ... (needs 72 % k == 0). --distribution=block:block is
# needed: the default is cyclic over sockets, which puts consecutive ranks on
# different Graces and all GPU ranks on GPU 0 (k=4 baselines run: 36x
# slower). --ntasks-per-socket must NOT be combined with it: the step then got
# cores of one Grace only and ranks were left unpinned (~50 ms MPI latency).
run() {
  local k=$1; shift
  local common="--ntasks=$((4 * k)) --cpus-per-task=$((72 / k)) --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
  echo
  echo "=============== k=$k: ./program $k $*"
  echo "placement:"
  srun $common bash -c 'echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V
  # drop the stack traces of the exit-time abort (static Kokkos), keep real errors
  timeout 400 srun $common ./program $k "$@" 2> >(grep -vE '^\[[^]]+\] \[ *[0-9]+\]|End of error message' >&2)
  wait
}

# args after k: degree n_refine switch_level n_subdivisions
# Q4, 24 cells/dir (912673 DoFs). Levels: 0..3 = Q1 h-levels, 4 = Q2, 5 = Q4 (fine);
# switch_level = finest level on the CPUs, 0 = full GPU hierarchy (regression).
# Baselines (job 2319072): bk3 0.0601 s, hybrid k=1/k=4 s=0 0.059-0.061 s per CG solve.
for k in 1 4 18; do
  for s in 0 1 2 3 4; do
    run $k 4 3 $s 3
  done
done
