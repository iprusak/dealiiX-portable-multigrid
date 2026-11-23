#!/bin/bash
#SBATCH --job-name=hybrid-gpu-mem
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=hybrid-gpu-mem-%j.out

# Where does the GPU memory of a hybrid run go? One node, p = 4, doubling mesh, 5 + 5:
#   mg-135, mg-270   poisson_cube_bk3, 1 rank per GPU, 135M and 270M DoFs per GPU
#   hy-135, hy-270   hybrid, K ranks per GPU, switch 8000; hy-270 is expected to run
#                    out of memory and then hang, so it gets a short timeout
# During every run nvidia-smi is sampled every SAMPLE_MS ms: memory per GPU, and
# memory per process on each GPU. Printed after the run: the peak per GPU, and per
# GPU the largest process (the GPU rank) and what the other processes hold (the
# CUDA contexts of the CPU-only ranks).
# With NVIDIA MPS, where the processes of a GPU share one context (default runs
# then: the two hybrid ones, and hy-270 gets the long timeout):
#   MPS=1 sbatch <this script>

REPO=${REPO:-$HOME/codes/dealiiX-coarse-levels-one-gpu}
DEAL_II=${DEAL_II:-/e/project1/dealii-x/ivan/sw/dealii/dealii-neon}
MG_BIN=${MG_BIN:-$REPO/performance_tests/poisson_cube_multigrid/build-neon/poisson_cube_bk3}
HYBRID_BIN=${HYBRID_BIN:-$REPO/performance_tests/poisson_cube_hybrid_gpu_cpu/build-neon/program}
K=${K:-18}
MPS=${MPS:-0}
SAMPLE_MS=${SAMPLE_MS:-200}
if [ "$MPS" = 1 ]; then
  RUNS=${RUNS:-"hy-135 hy-270"}
  HY270_TIMEOUT=${HY270_TIMEOUT:-2400}
else
  RUNS=${RUNS:-"mg-135 mg-270 hy-135 hy-270"}
  HY270_TIMEOUT=${HY270_TIMEOUT:-900}
fi
OUT=${OUT:-$PWD/hybrid-gpu-mem-${SLURM_JOB_ID:-local}}

source $REPO/scripts/jupiter/env.sh $DEAL_II
export NVCOMPILER_OMP_DISABLE_WARNINGS=true
export UCX_NUM_PPN=4

mkdir -p $OUT
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader

common="--mpi=pspmix --nodes=1 --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
gpu_only="$common --ntasks=4 --ntasks-per-node=4 --cpus-per-task=72"
hybrid="$common --ntasks=$((4 * K)) --ntasks-per-node=$((4 * K)) --cpus-per-task=$((72 / K))"

# one cycle: min and max size around the DoFs of the cycle
S135="538969625 538971625"
S270="1076889625 1076891625"

# run <label> <timeout in s> <srun options> <program and arguments>
run() {
  local label=$1 limit=$2 opts=$3; shift 3
  echo
  echo "=============== $label (MPS=$MPS, sampling $SAMPLE_MS ms): srun $opts $*"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -lms $SAMPLE_MS > $OUT/$label.gpu &
  local sampler_gpu=$!
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader,nounits -lms $SAMPLE_MS > $OUT/$label.apps &
  local sampler_apps=$!
  export HYBRID_LAUNCH_EPOCH=$(date +%s.%N)
  timeout $limit srun $opts "$@" | tee $OUT/$label.log
  kill $sampler_gpu $sampler_apps
  wait $sampler_gpu $sampler_apps 2> /dev/null
  echo "--- $label: peak per GPU"
  awk -F', ' '{if ($2 + 0 > m[$1]) m[$1] = $2 + 0} END {for (g in m) printf "GPU %s: %d MiB\n", g, m[g]}' $OUT/$label.gpu | sort
  echo "--- $label: processes per GPU (peak of each process)"
  awk -F', ' '{k = $1 " " $2; if ($3 + 0 > m[k]) m[k] = $3 + 0; name[k] = $4}
              END {for (k in m) {split(k, a, " "); g = a[1]; n[g]++; s[g] += m[k]; if (m[k] > b[g]) {b[g] = m[k]; top[g] = k}}
                   for (k in m) {split(k, a, " "); g = a[1]; if (k == top[g]) continue
                                 if (!(g in lo) || m[k] < lo[g]) lo[g] = m[k]; if (m[k] > hi[g]) hi[g] = m[k]}
                   for (g in n) printf "%s: %d processes, largest %d MiB (%s), the other %d together %d MiB (each %d to %d)\n",
                                       g, n[g], b[g], name[top[g]], n[g] - 1, s[g] - b[g], lo[g], hi[g]}' $OUT/$label.apps | sort
  echo "--- $label: process names on the GPUs"
  awk -F', ' '{names[$4]++} END {for (p in names) print "  " p}' $OUT/$label.apps | sort
}

if [ "$MPS" = 1 ]; then
  # control daemon for the four GPUs; the ranks find it through the pipe directory
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe-$SLURM_JOB_ID
  export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log-$SLURM_JOB_ID
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d
  mps_exit=$?
  echo "nvidia-cuda-mps-control -d: exit $mps_exit"
  if [ $mps_exit -ne 0 ]; then echo "MPS did not start, stopping"; exit 1; fi
fi

for r in $RUNS; do
  case $r in
    mg-135) run $r 2400 "$gpu_only" $MG_BIN 4 $S135 5 5 d 0 ;;
    mg-270) run $r 2400 "$gpu_only" $MG_BIN 4 $S270 5 5 d 0 ;;
    hy-135) run $r 2400 "$hybrid" $HYBRID_BIN $K 4 $S135 5 5 d 8000 0 ;;
    hy-270) run $r $HY270_TIMEOUT "$hybrid" $HYBRID_BIN $K 4 $S270 5 5 d 8000 0 ;;
    *) echo "unknown run $r" ;;
  esac
done

if [ "$MPS" = 1 ]; then
  echo
  echo "=============== MPS shutdown, end of the control log"
  echo quit | nvidia-cuda-mps-control
  tail -n 30 $CUDA_MPS_LOG_DIRECTORY/control.log
fi
