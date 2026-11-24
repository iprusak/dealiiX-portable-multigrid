#!/bin/bash
#SBATCH --job-name=elasticity-vmult
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=elasticity-vmult-%j.out

# Timings of the vmult variants of the elasticity operator
# (performance_tests/elasticity_operator: vmult_dealii, vmult_batched,
# vmult_fused, vmult_benchmark), degree + 1 points per direction, 4 ranks with
# one H100 each, on doubling meshes from 8M DoFs until a run fails (out of
# memory) or times out: the largest size per GPU is not known yet. One size per
# program run; the GPU memory is sampled every 2 s and its peak printed per run.
#
#   DEGREES="1 2 3 4"  degrees to run, split into several jobs if needed
#   MAX_DOFS=4.2e9     upper end of the ladder (total DoFs)
#   RUN_TIMEOUT=1800   seconds per program run (an OOM can hang the ranks)
#   REPS=20            repetitions per variant
#   BUILD=$HOME/codes/dealiiX-elasticity/build-elasticity-operator

BUILD=${BUILD:-$HOME/codes/dealiiX-elasticity/build-elasticity-operator}
DEGREES=${DEGREES:-"1 2 3 4"}
MAX_DOFS=${MAX_DOFS:-4.2e9}
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}
REPS=${REPS:-20}

cd "$(dirname "$BUILD")"
source scripts/jupiter/env.sh "${DEAL_II_DIR:-$WORK/dealii/dealii-master}"
export UCX_NUM_PPN=4

PROGRAM=$BUILD/program
ldd $PROGRAM | grep -E 'libdeal_II|libkokkoscore'

SAMPLES=$(mktemp)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 2 > "$SAMPLES" &
SAMPLER=$!

summary=""
for degree in $DEGREES; do
  dofs=8e6
  while python3 -c "import sys; sys.exit(0 if $dofs <= $MAX_DOFS else 1)"; do
    start_line=$(wc -l < "$SAMPLES")
    echo
    echo "=============== degree $degree, target $dofs DoFs"
    timeout $RUN_TIMEOUT srun --mpi=pspmix --ntasks=4 --cpus-per-task=72 --cpu-bind=cores \
      --gpu-bind=closest $PROGRAM $degree $dofs $REPS
    status=$?
    peak=$(tail -n +$((start_line + 1)) "$SAMPLES" | awk -F', ' '$2 > max { max = $2 } END { print max + 0 }')
    echo "peak GPU memory during the run: $peak MiB (of 97871)"
    summary="$summary\ndegree $degree target $dofs: exit $status, peak $peak MiB"
    if [ $status -ne 0 ]; then
      echo "degree $degree: run failed or timed out at $dofs DoFs, next degree"
      break
    fi
    dofs=$(python3 -c "print(2 * $dofs)")
  done
done

kill $SAMPLER
rm -f "$SAMPLES"

echo
echo "=============== summary"
echo -e "$summary"
