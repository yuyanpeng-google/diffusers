# Wan-AI/Wan2.2-I2V-A14B-Diffusers Recipe

1. Export the environment of GCP project
* Fill the PROJECT_ID and TPU_NAME
```
### 1. export env of gcp ###

export PROJECT_ID=<project_id>
export TPU_NAME=<tpu_name>
export ZONE=<zone>
export ACCELERATOR_TYPE=v6e-16
export RUNTIME_VERSION=v2-alpha-tpuv6e
```

2. Create the v6e-16 tpu vms on GCP
```
gcloud compute tpus tpu-vm create ${TPU_NAME}  \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --accelerator-type=${ACCELERATOR_TYPE}  \
  --version=${RUNTIME_VERSION} 
```

3. Prepare the python env on each tpu vms
```
### 3. prepare env on each host ###

run()
{
  local command=$1
  local worker=${2:-all}
  gcloud compute tpus tpu-vm ssh --zone "${ZONE}" "${ACCOUNT}@${TPU_NAME}" --project "${PROJECT_ID}" --worker=${worker} --command="$command"
}

BRANCH_NAME=wan2.2-main

SETUP_COMMAND="\
set -x && \
curl -LsSf https://astral.sh/uv/install.sh | sh && \
source ~/.local/bin/env && \
uv venv -p 3.12 && \
source .venv/bin/activate && \
git clone -b ${BRANCH_NAME} https://github.com/yuyanpeng-google/diffusers.git || true && \
cd diffusers && \
uv pip install -e . && \
uv pip install transformers accelerate && \
uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
uv pip install -U jax[tpu] && \
uv pip install torchax && \
uv pip install flax && \
uv pip install ftfy imageio imageio-ffmpeg && \
true
"

run "${SETUP_COMMAND}"
```

4. Run wan2.2 pipeline to generate the videos
```
### 4. run wan2.2 pipeline ###

run()
{
  local command=$1
  local worker=${2:-all}
  gcloud compute tpus tpu-vm ssh --zone "${ZONE}" "${ACCOUNT}@${TPU_NAME}" --project "${PROJECT_ID}" --worker=${worker} --command="$command"
}

BRANCH_NAME=wan2.2-main

RUN_COMMAND="\
set -x && \
source .venv/bin/activate && \
killall -9 python || true && \
sleep 10 && \
export JAX_COMPILATION_CACHE_DIR="/dev/shm/jax_cache" && \
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 && \
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 && \
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES='xla_gpu_per_fusion_autotune_cache_dir' && \
export HF_HUB_CACHE=/dev/shm/hf_cache && \
cd diffusers && \
git fetch && git reset --hard origin/${BRANCH_NAME} && \
cd exp && \
nohup python wan2p2_benchmark.py > $(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &
true
"
run "${RUN_COMMAND}"
```

5. See the results in stdout
```
...
output video done. 20251029_093753.mp4
Warmup and output video:  1961.571311s
...
Benchmark:  103.959559s
Done
```
Notice that the first time warmup need to compile the graph which is time consuming.

6. Use scp download generated videos
```
VIDEO_NAME=20251029_093753.mp4 # from the 5 stdout

gcloud compute tpus tpu-vm scp --zone "${ZONE}" "${TPU_NAME}:~/diffusers/exp/${VIDEO_NAME}" . --project "${PROJECT_ID}" --worker=0
```


# Install

Install dependencies, setup virtual env first if required.

Test use python 3.12

```sh
# install uv, python 3.12 and activate
curl -LsSf https://astral.sh/uv/install.sh | sh && \
source ~/.local/bin/env && \
uv venv -p 3.12 && \
source .venv/bin/activate && \
```

```sh
# install dependency
# pwd=.
uv pip install -e . && \
uv pip install transformers accelerate && \
uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
uv pip install -U jax[tpu] && \
uv pip install torchax && \
uv pip install flax && \
uv pip install ftfy imageio imageio-ffmpeg
```

To run:

```sh
# cwd=exp
python wan2p2_benchmark.py
```

### Result

```
# python wan2p2_benchmark.py
Benchmark:  103.959559s
Done
```

