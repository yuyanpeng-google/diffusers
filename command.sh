#!/bin/bash

set -ex

### 1. export env of gcp ###

export PROJECT_ID=<project_id>
export TPU_NAME=<tpu_name>
export ZONE=asia-northeast1-b
export ACCELERATOR_TYPE=v6e-16
export RUNTIME_VERSION=v2-alpha-tpuv6e

### 2. create vm ###

gcloud compute tpus tpu-vm create ${TPU_NAME}  \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --accelerator-type=${ACCELERATOR_TYPE}  \
  --version=${RUNTIME_VERSION} 

### 3. prepare env on each host ###

run()
{
  local command=$1
  local worker=${2:-all}
  gcloud compute tpus tpu-vm ssh --zone "${ZONE}" "${ACCOUNT}@${TPU_NAME}" --project "${PROJECT_ID}" --worker=${worker} --command="$command"
}

SETUP_COMMAND="\
set -x && \
sudo apt update && \
sudo apt install -y python3.10-venv && \
python -m venv venv && \
source venv/bin/activate && \
git clone -b wan2-1-141s https://github.com/yuyanpeng-google/diffusers.git || true && \
cd diffusers && \
git fetch origin && \
git reset --hard origin/wan2-1-141s && \
pip install -e . && \
sh -ex setup-dep.sh && \
true
"

run "${SETUP_COMMAND}"

### 4. run wan2.1 pipeline ###

run()
{
  local command=$1
  local worker=${2:-all}
  gcloud compute tpus tpu-vm ssh --zone "${ZONE}" "${ACCOUNT}@${TPU_NAME}" --project "${PROJECT_ID}" --worker=${worker} --command="$command"
}

RUN_COMMAND="\
set -x && \
source ~/venv/bin/activate && \
killall -9 python || true && \
sleep 10 && \
export JAX_COMPILATION_CACHE_DIR="/dev/shm/jax_cache" && \
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 && \
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 && \
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES='xla_gpu_per_fusion_autotune_cache_dir' && \
export HF_HUB_CACHE=/dev/shm/hf_cache && \
cd diffusers && \
git fetch && git reset --hard origin/wan2-1-141s && \
python wan_tx_splash_attn.py && \
true
"
run "${RUN_COMMAND}"

### 5. download generated video ###

VIDEO_NAME=<from_run_command_stdout>

gcloud compute tpus tpu-vm scp --zone "${ZONE}" "${TPU_NAME}:~/diffusers/${VIDEO_NAME}" . --project "${PROJECT_ID}" --worker=0
