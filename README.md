# vLLM vs SGLang Benchmark Comparison

A comprehensive benchmark comparison between vLLM and SGLang inference engines using the Qwen3-Coder-30B model.

---
## Youtube Video

https://www.youtube.com/watch?v=zNe6YfTOCRw&t=1s

## 📋 Overview

This repository contains benchmarking results and configurations for comparing vLLM and SGLang performance on the same hardware and model setup. The benchmarks use the ShareGPT dataset with 1000 prompts to evaluate throughput, latency, and resource utilization.

## 🖥️ System Requirements

### Hardware & OS
- **OS**: Linux (tested on Ubuntu 24.04)
  - Kernel: `6.14.0-29-generic #29~24.04.1-Ubuntu`
- **GPU**: NVIDIA GPU with CUDA support
- **Python**: 3.12+
- **Drivers** I use Driver Version:  590.48.01 that supports up to CUDA Version: 13.1 RTX PRO 6000 Blackwell Workstation Edition
### Prerequisites
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [Docker](https://docs.docker.com/engine/install/) with Docker Compose
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- HuggingFace account with access token

# 🚀 Quick Start

## 📁 Repository Structure

```text
.
├── Benchmarked vLLM vs SGLang So You Don't Have To - benchmark_comparison_neatly_categorized.csv.csv
├── docker-observability.yml
├── Inference_nvtop
│   ├── dynamo-trt-local_1.png
│   ├── dynamo-trt-local_2.png
│   ├── dynamo-trt_nats_etcd_1.png
│   ├── dynamo-trt_nats_etcd_4.png
│   ├── dynamo-vllm-local_1.png
│   ├── dynamo_vllm_local_2.png
│   ├── dynamo-vllm_nats_etcd_1.png
│   ├── dynamo-vllm_nats_etcd_4.png
│   ├── TRTLLM_inference.png
│   └── vLLM_Inference.png
├── Inference Single
│   ├── docker compose SGLang.yaml
│   ├── docker-compose TensorTRT.yaml
│   └── docker-compose vLLM.yaml
├── Local_Dockers_Without_Nats_Etcd
│   ├── docker_compose_D_TRT_Local_1.yaml
│   ├── docker_compose_ D_TRT_Local_2.yaml
│   ├── docker_compose_D_VLLM_Local_1.yaml
│   └── docker_compose_D_VLLM_Local_2.yaml
├── New_Benchmarks
│   ├── dynamo-trt-local_1_0214_1000_1024_1024.jsonl
│   ├── dynamo-trt-local_1_0214_1000_sharegpt.jsonl
│   ├── dynamo-trt-local_2_0214_1000_1024_1024.jsonl
│   ├── dynamo-trt-local_2_0214_1000_sharegpt.jsonl
│   ├── dynamo-trt_nats_etcd_1_0214_1000_1024_1024.jsonl
│   ├── dynamo-trt_nats_etcd_1_0214_1000_sharegpt.jsonl
│   ├── dynamo-trt_nats_etcd_4_0214_1000_1024_1024.jsonl
│   ├── dynamo-trt_nats_etcd_4_0214_1000_sharegpt.jsonl
│   ├── dynamo_vllm_local_2_0214_1000_1024_1024.jsonl
│   ├── dynamo_vllm_local_2_vllm_0214_1000_sharegpt.jsonl
│   ├── dynamo-vllm_nats_etcd_1_0214_1000_1024_1024.jsonl
│   ├── dynamo-vllm_nats_etcd_1_0214_1000_sharegpt.jsonl
│   ├── dynamo-vllm_nats_etcd_1_0215_1000_1024_1024.jsonl
│   ├── dynamo-vllm_nats_etcd_1_0215_1000_sharegpt.jsonl
│   ├── dynamo-vllm_nats_etcd_4_0214_1000_sharegpt.jsonl
│   ├── dynamo-vllm_nats_etcd_4_0215_1000_1024_1024.jsonl
│   ├── dynamo-vllm_nats_etcd_4_0215_1000_sharegpt.jsonl
│   ├── TRTLLM_inference_0215_1000_1024_1024.jsonl
│   ├── TRTLLM_inference_0215_1000_sharegpt.jsonl
│   ├── vllm_inference_single_0214_1000_1024_1024.jsonl
│   └── vllm_inference_single_0214_1000_sharegpt.jsonl
├── observability
│   ├── grafana_dashboards
│   │   ├── dcgm-metrics.json
│   │   ├── dynamo.json
│   │   ├── dynamo-operator.json
│   │   ├── kvbm.json
│   │   ├── sglang.json
│   │   └── temp-loki.json
│   ├── grafana-datasources.yml
│   ├── prometheus.yml
│   ├── tempo-datasource.yml
│   └── tempo.yaml
├── old benchmarks
│   ├── sglang_0128_1000_sharegpt.jsonl
│   └── vllm_0128_1000_sharegpt.jsonl
├── pyproject.toml
├── README.md
├── Server_Dockers_With_Nats_Etcd
│   ├── docker_compose_D_TRT_Nats_Etcd_1.yaml
│   ├── docker_compose_D_TRT_Nats_Etcd_4.yaml
│   ├── docker_compose_D_VLLM_Nats_Etcd_4.yaml
│   └── nats-server.conf
├── tests
│   ├── check_dynamo_health.py
│   ├── test_dynamo.py
│   ├── test_models_endpoint.py
│   └── verify_dynamo_trt.py
└── uv.lock
```

### 1. Environment Setup

```bash
# Just use ready pyproject.toml
uv sync

# I you want to create it yourself.
uv venv sglang_env --python 3.12
source sglang_env/bin/activate
uv pip install sglang
```

### 2. Configure Environment Variables

Create a `.env` file in the project root or rename env_show to .env and fill your key

```bash
HF_TOKEN=your_token_here
```

### 2.1 Hugging Face Cache And Fast Download

Docker services mount:
`${HOME}/.cache/huggingface:/data/hf`

This path is used as the shared local model repository/cache for all containers. It's default of hugging face for convenience.

For faster model downloads with `hf-transfer`:

```bash
export HF_TOKEN=$HF_TOKEN # this takes token from the .env file 
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3-32B-FP8
# or
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3-32B-FP8 export HF_TOKEN=$HF_TOKEN
```

If `HF_TOKEN` is already loaded in your shell or from `.env`, you can also do:

```bash
export HF_TOKEN=$HF_TOKEN
```

### 3. Create KV Folder Once (for `Local_Dockers_Without_Nats_Etcd` dockers)

```bash
mkdir -p ~/.cache/dynamo_kv
chmod -R a+rwX ~/.cache/dynamo_kv
```

## 🐳 Docker Compose Commands

All commands below should be run from the repository root:
`/home/luke/Documents/Code/VLLM_SGLang_DYNAMO`

```bash
# Inference Single
docker compose -f "Inference Single/docker-compose vLLM.yaml" up -d
docker compose -f "Inference Single/docker compose SGLang.yaml" up -d

# Local_Dockers_Without_Nats_Etcd
docker compose -f "Local_Dockers_Without_Nats_Etcd/docker_compose_D_TRT_Local_1.yaml" up -d
docker compose -f "Local_Dockers_Without_Nats_Etcd/docker_compose_ D_TRT_Local_2.yaml" up -d
docker compose -f "Local_Dockers_Without_Nats_Etcd/docker_compose_D_VLLM_Local_1.yaml" up -d
docker compose -f "Local_Dockers_Without_Nats_Etcd/docker_compose_D_VLLM_Local_2.yaml" up -d

# Server_Dockers_With_Nats_Etcd
docker compose -f "Server_Dockers_With_Nats_Etcd/docker_compose_D_TRT_Nats_Etcd_1.yaml" up -d
docker compose -f "Server_Dockers_With_Nats_Etcd/docker_compose_D_TRT_Nats_Etcd_4.yaml" up -d
docker compose -f "Server_Dockers_With_Nats_Etcd/docker_compose_D_VLLM_Nats_Etcd_4.yaml" up -d

# Observability stack
docker compose -f "docker-observability.yml" up -d
```

### Server Compose: Aggregated vs Prefill/Decode Split

`Server_Dockers_With_Nats_Etcd/docker_compose_D_TRT_Nats_Etcd_4.yaml` and
`Server_Dockers_With_Nats_Etcd/docker_compose_D_VLLM_Nats_Etcd_4.yaml`
currently run one backend worker with `gpus: all`.

That is aggregated serving (single worker handles both prefill + decode), not
prefill/decode-separated serving.

If you want prefill/decode separation, keep one shared `nats-server`,
`etcd-server`, and `dynamo-frontend`, then run two backend services and pin
each one to different GPUs.

Example (vLLM split):

```yaml
services:
  dynamo-vllm-prefill:
    image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1
    command: python3 -m dynamo.vllm --model Qwen/Qwen3-32B-FP8 --is-prefill-worker --gpu-memory-utilization 0.8 --max-model-len 10000
    environment:
      - ETCD_ENDPOINTS=etcd-server:2379
      - NATS_SERVER=nats://nats-server:4222
      - NVIDIA_VISIBLE_DEVICES=0
    gpus: all
    ipc: host

  dynamo-vllm-decode:
    image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1
    command: python3 -m dynamo.vllm --model Qwen/Qwen3-32B-FP8 --gpu-memory-utilization 0.8 --max-model-len 10000
    environment:
      - ETCD_ENDPOINTS=etcd-server:2379
      - NATS_SERVER=nats://nats-server:4222
      - NVIDIA_VISIBLE_DEVICES=1
    gpus: all
    ipc: host
```

Copy the same `volumes`, token env vars, and `ulimits` from your existing
single backend service into both split services.

For TensorRT-LLM split mode, use the same two-service pattern and set:

```bash
# prefill worker
python3 -m dynamo.trtllm ... --disaggregation-mode prefill

# decode worker
python3 -m dynamo.trtllm ... --disaggregation-mode decode
```

Multi-GPU mapping examples:

- 2 GPUs total: prefill `NVIDIA_VISIBLE_DEVICES=0`, decode `=1`
- 4 GPUs total: prefill `=0,1`, decode `=2,3`
- 8 GPUs total: prefill `=0,1,2`, decode `=3,4,5,6,7`

Start balanced (50/50 split), then give more GPUs to decode if output lengths
are long.

Useful commands:

```bash
docker compose -f "Inference Single/docker-compose vLLM.yaml" logs -f
docker compose -f "Inference Single/docker compose SGLang.yaml" logs -f
docker compose -f "Inference Single/docker-compose vLLM.yaml" down
docker compose -f "Inference Single/docker compose SGLang.yaml" down
```

### vLLM Image Note (`vLLM_Inference`)

When running `vLLM_Inference` on host NVIDIA driver `590.48.01` (CUDA `13.1`), you may hit:
`RuntimeError: Unexpected error from cudaGetDeviceCount() ... Error 803: system has unsupported display driver / cuda driver combination`.

This is caused by a CUDA compatibility conflict between the container image and newer host drivers (580+).
The workaround used in this repo is the extra mount in
`Inference Single/docker-compose vLLM.yaml`:

```yaml
volumes:
  - /dev/null:/etc/ld.so.conf.d/00-cuda-compat.conf
```

## Dynamo Run Modes (Docs + This Repo Mapping)

Use this section when running Dynamo manually (without compose) and to match behavior in your Docker files.

### Required Token Variables

For this repo, keep `HF_TOKEN` in `.env`, and pass it to the container as `HUGGING_FACE_HUB_TOKEN`.

```bash
export HF_TOKEN=your_token_here
```

When using `docker run`, pass both to avoid ambiguity:

```bash
-e HF_TOKEN="$HF_TOKEN" \
-e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
```

### Option A: Containers (Recommended by Dynamo)

These images include dependencies:

```bash
# SGLang
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1

# TensorRT-LLM
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1

# vLLM
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1
```

For TensorRT-LLM with Hugging Face auth + persistent model cache (recommended in this repo):

```bash
docker run --gpus all --network host --rm -it \
  -e HF_TOKEN="$HF_TOKEN" \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  -e HF_HOME=/data/hf \
  -e HF_HUB_CACHE=/data/hf/hub \
  -v ${HOME}/.cache/huggingface:/data/hf \
  nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1
```

If you want the exact doc-style cache mount instead:

```bash
-v ${HOME}/.cache/huggingface:/root/.cache/huggingface
```

### A) One-Container Quick Start (NATS + etcd)

This matches `Server_Dockers_With_Nats_Etcd/*`.

1. Start infra:

```bash
nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://0.0.0.0:2379 \
     --data-dir /tmp/etcd &
```

2. Export endpoints:

```bash
export ETCD_ENDPOINTS=localhost:2379
export NATS_SERVER=nats://localhost:4222
```

3. Start frontend:

```bash
python3 -m dynamo.frontend --http-port 8000
```

4. Start backend worker (choose one):

TensorRT-LLM:

```bash
python3 -m dynamo.trtllm \
  --model-path Qwen/Qwen3-32B-FP8 \
  --free-gpu-memory-fraction 0.80 \
  --max-seq-len 10000
```

vLLM:

```bash
python3 -m dynamo.vllm \
  --model Qwen/Qwen3-32B-FP8 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 10000
```

### B) Local Mode Without NATS/etcd (`--store-kv file`)

This matches `Local_Dockers_Without_Nats_Etcd/*`.
In Dynamo docs this is often shown as `--discovery-backend file`; in your current Docker files/runtime, use `--store-kv file`.

Use these flags/env in both frontend and backend:

```bash
--store-kv file
```

```bash
export DYN_REQUEST_PLANE=tcp
export DYN_FILE_KV=/data/dynamo_kv
```

And mount:

```bash
-v ${HOME}/.cache/dynamo_kv:/data/dynamo_kv
```

### C) `Inference Single/docker-compose TensorTRT.yaml` (Not Dynamo)

This file runs pure TensorRT-LLM server (`trtllm-serve`), so do not pass NATS/etcd vars there.
If model download/auth is needed, add:

```yaml
environment:
  - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}
```

Or run compose with env file explicitly:

```bash
docker compose --env-file .env -f "Inference Single/docker-compose TensorTRT.yaml" up -d
```

### D) Send a Request (OpenAI-compatible API)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role":"user","content":"Say hello in one sentence."}],
    "stream": false
  }' | jq
```

For streaming tokens:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role":"user","content":"Say hello in one sentence."}],
    "stream": true
  }'
```

## 📊 Running Benchmarks

### Dataset Types

**ShareGPT**

is a conversational dataset consisting of human-AI chat logs, often
used to simulate real user behavior in, for example, API-based chatting.

**Random**

is a synthetically generated dataset that creates random token
sequences to stress-test the model with configurable input/output
lengths and distributions.

**Wait times/Data Distribution:** `sharegpt` naturally has a high variance in prompt lengths, while `random` allows for a uniform distribution.

| Feature | `sharegpt` | `random` |
| --- | --- | --- |
| **Data Source** | Real-world, cleaned conversations | Synthetically generated tokens |
| **Turn Structure** | Multi-turn, varied lengths (often uses first 2 turns) | Configurable input/output length |
| **Best For** | Realistic, varied workload simulation | Throughput/latency testing with specific, uniform workloads |
| **Complexity** | High variability in request sizes | Uniform or predictable, allowing focused testing |
| **Default** | Often the default in older benchmarks | Often the default in newer benchmarks |
| **Primary Goal** | Measure **Theoretical Peak** performance. | Measure **Production** performance. |

```bash
python3 -m sglang.bench_serving \
  --backend vllm \
  --base-url http://0.0.0.0:8000 \
  --model Qwen/Qwen3-32B-FP8 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --num-prompts 1000 \
  --request-rate inf \
  --max-concurrency 512 \
  --warmup-requests 10

python3 -m sglang.bench_serving \
  --backend vllm \
  --base-url http://0.0.0.0:8000 \
  --model Qwen/Qwen3-32B-FP8 \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --request-rate inf \
  --max-concurrency 512 \
  --warmup-requests 10
```

## How to Compare LLMs (Practical Guide)

This is a lightweight process for comparing multiple LLMs in a way that is **fair**, **repeatable**, and **useful for choosing a model that fits your needs**.

### 1) Start with a prompt dataset

Pick (or build) a dataset of prompts that matches your real use case.

- **Use your own prompts** if possible (most representative).
- If you need a starting point:
  - ShareGPT-style conversation datasets
  - Random or curated prompt sets on Hugging Face Datasets: https://huggingface.co/datasets

Tip: Include a mix of:

- common "happy path" prompts
- edge cases (ambiguous questions, missing info, contradictory constraints)
- "must not fail" prompts (critical business flows)

### 2) Shortlist candidate models using benchmarks + community signal

Benchmarks and other people's evaluations are great for narrowing down options, but they **don't guarantee** the best model for *your* use case.

Where to look:

- Hugging Face model listings: https://huggingface.co/models
- Community discussions (often more practical than benchmarks):
  - LocalLLaMA subreddit: https://www.reddit.com/r/LocalLLaMA/

If you can't host models yourself, a common hosted option is:

- OpenRouter: https://openrouter.ai/

### 3) Check deployment compatibility (engines, resources, context)

Before you invest time, confirm the model is actually deployable in your environment.

#### Availability on your engine(s)

- Confirm the model is supported on the platforms you'll actually use:
  - local inference (e.g., llama.cpp / vLLM / TensorRT-LLM, etc.)
  - cloud/API provider(s)
  - routing/aggregation services (if you use them)

#### Resource requirements

- Ensure you have enough resources for your **target context length** and throughput needs.
- Different engines have different memory footprints and constraints:
  - GPU VRAM (often the limiting factor)
  - CPU RAM (can matter a lot for some setups)
  - disk size / model format requirements (GGUF, safetensors, etc.)

**Goal:** avoid "winner models" that you can't deploy.

### 4) Validate scaling + async support (throughput, batching, quantization, cost)

Put this **right after deployment compatibility**, because it's part of "can I run this in production?"

If you need async handling and scaling, check:

#### Can your engine scale the way you need?

- **Concurrent requests**: does the engine support async serving / multiple in-flight requests?
- **Batching**: can it batch requests efficiently (continuous batching, dynamic batching)?
- **Streaming**: does it support streaming tokens while still handling concurrency?

#### Does your chosen quantization allow it?

Quantization can change what's possible:

- Some quant formats may limit:
  - GPU acceleration / kernel support
  - batching efficiency
  - maximum context or KV cache behavior (engine-dependent)
- Verify that your quant choice is supported by your serving engine *for your target hardware*.

#### What does it cost at scale?

Estimate (even roughly):

- cost per 1M tokens (input + output)
- cost per successful request (if retries/failures happen)
- infra cost (GPU hours, memory, autoscaling overhead)
- latency targets (p50/p95) and how cost changes when you chase lower latency

**Practical check:** run a load test with representative traffic (even a small one) and record:

- throughput (req/s), tokens/s
- p50/p95 latency
- GPU/CPU/RAM utilization
- failure rate (OOM, timeouts)

### 5) Normalize settings so the comparison is fair

Try to keep generation parameters as similar as possible across models.

Check and align (when possible):

- temperature
- top_p / top_k
- max_tokens
- system prompt / formatting instructions
- tool use (on/off), retrieval (on/off)

If you can't align everything (because some platforms hide defaults), decide one approach and stick to it:

- **Option A: "Defaults-only"** (realistic usage, less controlled)
- **Option B: "Normalized settings"** (more scientific, more fair)

Write down what you used so results are reproducible.

### 6) Compare outputs using a simple rubric (not vibes)

For each prompt, score models on the dimensions that actually matter to you, for example:

- **Correctness / factuality**
- **Instruction following** (did it obey constraints?)
- **Completeness** (did it answer all parts?)
- **Clarity / structure**
- **Hallucinations** (made-up info, fake citations)
- **Latency + cost** (if relevant)

Tip: Run each prompt multiple times if your settings are stochastic (e.g., temperature > 0), variability matters.

### 7) Note the deployment reality differences

One big practical difference:

- **Local models**: behavior stays stable unless *you* change the model/version.
- **API-hosted models**: behavior can change over time (providers may update models without much notice), which can lead to unexpected regressions.

If stability matters, track:

- model name/version (or release date)
- provider/model routing details (if applicable)

### Recommended deliverable: a 1-page comparison summary

Include:

- the dataset used (and size)
- platforms/engines tested
- parameter settings
- a small table: model -> scores + notes
- top 1-2 failure modes per model (most actionable part)
- scaling notes: throughput, latency (p50/p95), and cost ballpark

## Resources

- **GitHub Repo**: [vLLM vs SGLang Benchmarks](https://github.com/lukaLLM/vLLM_vs_SGLang_benchmarks)
- **vLLM Documentation**: 
  - [vLLM GitHub](https://github.com/vllm-project/vllm)
  - [vLLM Docs](https://docs.vllm.ai)
- **SGLang Documentation**:
  - [SGLang GitHub](https://github.com/sgl-project/sglang)
  - [SGLang Docs](https://docs.sglang.io/references/learn_more.html)
- **Benchmark Tool**: [SGLang bench_serving](https://docs.sglang.io/developer_guide/bench_serving.html)
- **NVIDIA Dynamo**: [Dynamo GitHub](https://github.com/ai-dynamo/dynamo)
- **TensorRT-LLM**: [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- **AI Perf**: [ai-dynamo/aiperf GitHub](https://github.com/ai-dynamo/aiperf)
- **Paper**: [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

## �📄 License

This benchmark project is for educational and comparison purposes.
