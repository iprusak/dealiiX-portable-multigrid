#!/bin/bash
#SBATCH --job-name=hybrid
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
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

# k ranks per Grace (= per GPU), 4k ranks on the node. Ranks of a block must
# share a socket -> --ntasks-per-socket=k; the layout table printed by the
# program shows cpu/socket/device per rank.
run() {
  local k=$1; shift
  echo "=============== k=$k: ./program $k $*"
  srun --ntasks=$((4 * k)) --ntasks-per-socket=$k --distribution=block:block \
       --cpu-bind=cores --gpu-bind=closest --mem-bind=local ./program $k "$@"
}

# args after k: degree n_refine switch_level n_subdivisions

# 1. regression: switch_level=0 = full GPU hierarchy, CG its must equal poisson_cube_bk3
run 1 4 3 0 3

# 2. hybrid, 4 ranks: GPU on the fine levels, Grace on the coarse ones
run 1 4 3 2 3

# 3. one GPU rank + 3 CPU-only ranks per GPU
run 4 4 3 2 3

# 4. full node, 18 ranks per Grace; sweep the switch level
for s in 0 1 2 3; do
  run 18 4 4 $s 3
done
