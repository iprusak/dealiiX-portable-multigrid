#!/bin/bash
#SBATCH --job-name=hybrid-dbl
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=hybrid-dbl-%j.out

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
placement_k=""
run() {
  local k=$1; shift
  local common="--ntasks=$((4 * k)) --cpus-per-task=$((72 / k)) --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
  echo
  echo "=============== k=$k: ./program $k $*"
  if [ "$k" != "$placement_k" ]; then
    placement_k=$k
    echo "placement:"
    srun $common bash -c 'echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V
  fi
  # drop the stack traces of the exit-time abort (static Kokkos), keep real errors
  timeout 900 srun $common ./program $k "$@" 2> >(grep -vE '^\[[^]]+\] \[ *[0-9]+\]|End of error message' >&2)
  wait
}

# The usual doubling-mesh test: Q4, Chebyshev degree 5 (5 pre + 5 post), cycle c of
# poisson_cube_bk3's sequence: n_refine = c / 3, stretch = c % 3, coarse cells 2^stretch.
# Compare switch_level=0 (full GPU hierarchy) with the GPU-only run of poisson_cube_bk3
#   ./program 4 <min> <max> 5 5 d 0
# args after k: degree n_refine switch_level n_subdivisions(unused) n_smooth stretch
#
# Cells of Q1 level l: 2^stretch * 8^l. The CPU switch levels tried are the finest
# level with at most ~2200 cells (best at k=1 on the 24^3 mesh: 1728 cells) and with
# at most ~20000 (13824 cells, was slower with one scalar core per rank).
CELLS_A=2200
CELLS_B=20000
N_SMOOTH=5

# finest Q1 level with at most $2 cells on the coarse grid with $1 cells
finest_level() {
  local l=0 cells=$1
  while [ $((cells * 8)) -le $2 ]; do cells=$((cells * 8)); l=$((l + 1)); done
  echo $l
}

for k in 4 18; do
  for c in 12 13 14 15 16 17 18 19 20; do
    nr=$((c / 3)); st=$((c % 3)); cells0=$((1 << st))
    sa=$(finest_level $cells0 $CELLS_A); sb=$(finest_level $cells0 $CELLS_B)
    seen=" "
    for s in 0 $sa $sb; do
      case "$seen" in *" $s "*) continue ;; esac
      seen="$seen$s "
      [ $s -le $nr ] || continue
      echo "######## cycle $c: n_refine=$nr stretch=$st switch_level=$s"
      run $k 4 $nr $s 3 $N_SMOOTH $st
    done
  done
done
