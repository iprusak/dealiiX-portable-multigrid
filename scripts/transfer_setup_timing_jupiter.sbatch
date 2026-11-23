#!/bin/bash
#SBATCH --job-name=mg-transfer-setup
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=288
#SBATCH --cpus-per-task=1
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=mg-transfer-setup-%j.out

# TEMPORARY (transfer setup timing, not for main): where does "Setup transfers" go?
# poisson_cube_bk3 built from the instrumented worktree prints per transfer level
#   TWT geo-weights / poly-weights: the five weight steps (alloc, count,
#       compress+invert+ghost, gather, deep_copy), polynomial also the shared index
#       pass, finalize and the two index blocks
#   TWT geo-reinit / poly-reinit: the whole reinit(), geometric with its sections
# as the maximum over the ranks. TWT_STOP=1 skips smoothers and solve, every cycle
# of the size range is set up. Doubling mesh, float V-cycle, 1 rank per GPU.
# Sizes: MIN_PER_GPU .. MAX_PER_GPU DoFs per GPU (default up to 270M, the largest
# that fits for p=4; p=2 at that size is untested and may run out of memory, the
# smaller cycles are printed before).
#   REPO=<checkout> BIN=<build dir>/poisson_cube_bk3 [DEAL_II_DIR=<install>] sbatch <this script>
# DEAL_II_DIR: the install the binary was built with (default of scripts/jupiter/env.sh).

REPO=${REPO:?set REPO to the instrumented checkout}
BIN=${BIN:?set BIN to its poisson_cube_bk3}
DEGREES=${DEGREES:-"2 4"}
GPU_COUNTS=${GPU_COUNTS:-"1 4"}
MIN_PER_GPU=${MIN_PER_GPU:-1000000}
MAX_PER_GPU=${MAX_PER_GPU:-271000000}

source $REPO/scripts/jupiter/env.sh $DEAL_II_DIR
export NVCOMPILER_OMP_DISABLE_WARNINGS=true
export TWT_STOP=1

ldd $BIN | grep -E 'libdeal_II|libkokkoscore'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

OUT=$PWD/mg-transfer-setup-$SLURM_JOB_ID
mkdir -p $OUT

for p in $DEGREES; do
  for gpus in $GPU_COUNTS; do
    label=p${p}_gpus${gpus}
    opts="--mpi=pspmix --ntasks=$gpus --cpus-per-task=72 --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
    echo
    echo "=============== $label: srun $opts $BIN $p $((MIN_PER_GPU * gpus)) $((MAX_PER_GPU * gpus)) 5 5 d 0"
    timeout 1500 srun $opts $BIN $p $((MIN_PER_GPU * gpus)) $((MAX_PER_GPU * gpus)) 5 5 d 0 2>&1 | tee $OUT/$label.out
  done
done

echo
echo "######## rank counts (must be the number of GPUs) and phase times"
grep -H -E "Number of MPI ranks|Number of degrees of freedom|Setup transfers" $OUT/*.out
