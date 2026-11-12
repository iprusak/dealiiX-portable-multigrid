#!/bin/bash
#SBATCH --job-name=hybrid-sweep
#SBATCH --account=dealii-X
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --cpus-per-task=4
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=hybrid-%j.out

module --force purge
module load Stages/2026 nvidia-compilers OpenMPI FlexiBLAS

# dealii-neon: NEON on the Grace side, shared Kokkos 5.2.2 (clean exit)
export LD_LIBRARY_PATH=/e/project1/dealii-x/ivan/sw/dealii/dealii-neon/lib:/e/project1/dealii-x/ivan/sw/kokkos/kokkos-5.2.2/lib64:$WORK/sw/lib:$WORK/sw/lib64:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=1
export KOKKOS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NVCOMPILER_OMP_DISABLE_WARNINGS=true

ldd ./program | grep -E 'libdeal_II|libkokkoscore'
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
  if [ $((72 % k)) -ne 0 ]; then echo "k=$k does not divide the 72 cores of a Grace, skipped"; return; fi
  local common="--mpi=pspmix --ntasks=$((4 * k)) --cpus-per-task=$((72 / k)) --distribution=block:block --cpu-bind=cores --gpu-bind=closest --mem-bind=local"
  echo
  echo "=============== k=$k: ./program $k $*"
  echo "placement:"
  srun $common bash -c 'echo "rank $SLURM_PROCID $(hostname) gpu=$CUDA_VISIBLE_DEVICES $(grep Cpus_allowed_list /proc/self/status)"' | sort -V
  timeout 400 srun $common ./program $k "$@"
}

# args after k: degree minsize maxsize n_pre n_post mesh switch_dofs
# Q4, 24 cells/dir (912673 DoFs). Levels and their DoFs: 0..3 = Q1 h-levels (64, 343,
# 2197, 15625), 4 = Q2 (117649), 5 = Q4 (fine, 912673). The switch level is the finest
# level with at most switch_dofs DoFs per rank; here switch_dofs is set to select each
# level in turn: 0 = full GPU hierarchy (regression), the fine level = CPU only.
# 5 pre + 5 post smoothing steps (the October sweep, job 2320320, used 3 + 3).
# k has to divide 72 (72/k cores per rank): 4k = 4 .. 288 ranks on the node.
for k in 1 4 12 18 24 36 72; do
  ranks=$((4 * k))
  for level_dofs in 0 343 2197 15625 117649 912673; do
    run $k 4 900000 1000000 5 5 s $(((level_dofs + ranks - 1) / ranks))
  done
done
