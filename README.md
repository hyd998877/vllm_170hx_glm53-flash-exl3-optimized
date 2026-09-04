# GLM-5.3-Flash EXL3 optimized vLLM for CMP 170HX / SM80

面向 **4× NVIDIA CMP 170HX（SM80）** 的 GLM-5.3-Flash 长上下文推理优化版。
本仓库在 vLLM 的 GLM-5.3 支持基础上增加 EXL3 读取、Marlin INT4
sidecar、SM80 稀疏 MLA、DFlash2 投机解码以及 PP4 调度优化，提供
OpenAI 兼容接口、512K 请求上限和多模态输入。

> This is an unofficial, hardware-specific research fork. It is not an
> official vLLM or Z.ai release. The English quick summary is available
> [below](#english-summary).

## 当前状态

- 目标模型：[`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
- 投机模型：[`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
- 并行方式：TP1 / PP4，每张 GPU 一个 pipeline stage
- 上下文上限：524,288 tokens
- 服务接口：OpenAI-compatible Chat Completions
- 图像输入：支持；默认 `multimodal` profile 会加载视觉编码器并预留激活显存
- 当前正式 profile：`13,12,11,9 + DFlash2 k=2 + pairpack + adaptive prefill + GMU 0.970`

该项目针对特定 checkpoint、硬件和负载优化，不应理解为对所有模型或所有请求
形状都更快。

并发、PP/TP、DFlash、KV 和 kernel 优化采用逐级淘汰方案，避免对每个候选重复
运行耗时的完整 128K 测试。已执行的门槛、原始结果和淘汰结论见
[OPTIMIZATION_EXECUTION_PLAN.md](OPTIMIZATION_EXECUTION_PLAN.md)。

## 主要改进

- GLM-5.3 routed experts 的选择性 EXL3 K4/MCG reader。
- 将第 3–44 层 routed experts 离线转换为 group-size 64 的 Marlin INT4
  sidecar；在 SM80 decode 上替换较慢的原始 EXL3 路径。
- 非 routed 线性层的受控 FP8 weight-only/Marlin 路径，范围与目标 checkpoint
  的官方 native policy 对齐。
- SM80 Triton sparse-MLA backend、top-k/indexer、混合 Mamba/MLA KV cache
  兼容与 KV 账本诊断。
- PP4 自定义分层、异步 pipeline hand-off、pairpack decode phase，以及仅用于
  固定并发基准的 prefill cohort barrier。
- 自适应长 prefill：唯一冷请求使用大块，并发或 decode 到来后自动恢复小块，
  避免用一个全局阈值同时牺牲 TTFT 和在线 ITL。
- DFlash2 在 GLM NoPE/auxiliary RoPE 布局下的接受率修复、fused context-KV
  projection 和运行期 JIT warmup。
- CUDA Graph `FULL_DECODE_ONLY`，针对 3/6/9/12/15/18 token batch 捕获。
- 多模态 chat template、工具调用和 reasoning parser 支持。

## 已验证性能

测试机为 4× CMP 170HX，每卡 65,344 MiB、SM80、200 W 默认功耗上限；主机
内存 251 GiB。所有六路正式结果均使用独立的 131,072-token prompt、温度 0，
并检查 TTFT 同步性和 `Waiting=0`。

| Profile | 工作负载 | 聚合 decode | 平均每路 decode | 最低每路 | 说明 |
|---|---:|---:|---:|---:|---|
| 当前多模态正式配置 | 6×128K → 512 | **230.24 tok/s** | 45.90 tok/s | 38.58 tok/s | 修正共同窗口；TTFT spread 0.102 s；无 Waiting/Deferred/Preemption |
| 文本专用热测最佳 | 6×128K → 512 | **222.48 tok/s** | 40.29 tok/s | 37.08 tok/s | `13,11,11,10`，六路 TTFT 差异约 0.20 s |
| 文本专用长输出 | 6×128K → 8192 | **212.95 tok/s** | 37.99 tok/s | 35.47 tok/s | 全程无 Waiting/Deferred |
| 多模态正式配置 | 6×128K → 512 | **208.01 tok/s** | 36.20 tok/s | 34.46 tok/s | `13,12,11,9`，视觉 profiling 开启 |

当前 230.24 tok/s 使用从最早首 token 到最晚末 token 的修正共同窗口；历史
222.48 tok/s 使用旧热测/last-TTFT 口径，两者不构成严格配对 A/B。这里的
“聚合 decode”不是单路速度，也不包含 128K prefill；历史 222.48 那次端到端聚合为
11.04 tok/s，TTFT 约 264 秒。目录中曾出现 242–279 tok/s 的数值，但这些样本
存在第六路延迟入场、Waiting/Deferred 或错峰 TTFT，因此没有作为正式最佳值。

性能结果只代表上述硬件、模型、sampling 和测试口径。DFlash 接受率与文本类型
有关，短回答还可能被首次 Triton JIT 或 CUDA Graph capture 放大延迟。

## 硬件与磁盘要求

推荐按已验证配置准备：

- Linux x86-64；Python 3.12。
- 4 张 compute capability 8.0 GPU，每卡至少约 64 GiB 显存，GPU 间 NCCL
  通信正常。
- 至少 250 GiB 主机内存。模型加载阶段会出现较高 CPU 和 page-cache 使用量。
- 约 340 GiB 可用磁盘：目标 checkpoint 约 164 GiB、Marlin sidecar 约
  151 GiB、DFlash2 约 4.5 GiB，另需编译和缓存空间。
- 可用的 CUDA toolchain、C++20 compiler 和 Ninja。已验证环境使用 NVIDIA
  driver 610.43.02、CUDA 13.2、PyTorch 2.13.0+cu132、Triton 3.7.1、
  FlashInfer 0.6.15.post1 和 Transformers 5.15.0。

未验证较小显存、不同 SM 架构、TP>1 或少于四张 GPU 的组合。本项目不会主动
修改 GPU 功耗上限或时钟。

## 部署

### 1. 获取代码并创建环境

```bash
git clone https://github.com/hyd998877/vllm_170hx_glm53-flash-exl3-optimized.git
cd vllm_170hx_glm53-flash-exl3-optimized

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements/build.txt
uv pip install --no-build-isolation -e .
uv pip install --no-deps exllamav3==0.0.43
```

vLLM/CUDA 的可用 wheel 与本机驱动强相关。如果以上源码构建失败，请先按
[vLLM GPU 源码安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
安装与本机 CUDA ABI 对应的 PyTorch 和构建依赖，再执行 editable install。
不要将其他 vLLM checkout 的旧 `.so` 直接复制到本仓库。

### 2. 下载模型

```bash
hf download brandonmusic/GLM-5.3-Flash-tr3-4bpw \
  --local-dir /models/GLM-5.3-Flash-tr3-4bpw

hf download incoai/GLM-5.3-Flash-DFlash2 \
  --local-dir /models/GLM-5.3-Flash-DFlash2
```

模型权重不包含在本仓库。目标 EXL3 checkpoint 使用其模型仓库中的
ShapleyMCG 许可；DFlash2 使用 CC BY-NC-ND 4.0，仅面向其许可允许的研究和
评估场景。部署前必须自行检查两者的最新许可。

### 3. 编译 ExLlamaV3 扩展

生产配置使用 stock ExLlamaV3 0.0.43 native extension：

```bash
export TORCH_CUDA_ARCH_LIST=8.0
export TORCH_EXTENSIONS_DIR=/models/.torch_extensions

.venv/bin/python scripts/build_exllamav3_ext.py \
  --source stock \
  --build-dir "$TORCH_EXTENSIONS_DIR/exllamav3_ext"
```

构建完成后应存在：

```text
/models/.torch_extensions/exllamav3_ext/exllamav3_ext.so
```

`--source shared-had` 会构建本仓库的实验性 fused/shared-Hadamard 扩展。它没有
进入当前 230.24 tok/s 正式配置，部署时不要默认启用。

### 4. 生成 Marlin INT4 sidecar

最快配置需要第 3–44 层的 Marlin sidecar。转换会逐 expert 重建 EXL3 权重并
重新量化；输出约 151 GiB。下面使用四张卡并行转换不相交的层：

```bash
export MODEL=/models/GLM-5.3-Flash-tr3-4bpw
export MARLIN_DIR=/models/GLM-5.3-Flash-tr3-4bpw-marlin-int4-gs64
export TORCH_EXTENSIONS_DIR=/models/.torch_extensions
export VLLM_EXL3_EXTENSION_DIR="$TORCH_EXTENSIONS_DIR/exllamav3_ext"

for gpu in 0 1 2 3; do
  (
    for ((layer=3+gpu; layer<=44; layer+=4)); do
      CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python \
        scripts/convert_exl3_to_marlin.py \
        --model "$MODEL" \
        --out-dir "$MARLIN_DIR" \
        --layer "$layer" \
        --group-size 64 \
        --device 0
    done
  ) &
done
wait
```

转换器使用临时文件并在完成后原子发布，进程中断不会把半成品误认为有效
sidecar。正式启动脚本还会检查 42 个文件是否全部存在且非空。

可抽样比较 sidecar 与源 EXL3 层：

```bash
.venv/bin/python benchmarks/kernels/validate_exl3_marlin_layer.py \
  --model "$MODEL" \
  --extension "$VLLM_EXL3_EXTENSION_DIR" \
  --layer 3 \
  --sidecar "$MARLIN_DIR/layer-03.safetensors"
```

### 5. 启动服务

多模态生产 profile：

```bash
MODEL=/models/GLM-5.3-Flash-tr3-4bpw \
DFLASH_MODEL=/models/GLM-5.3-Flash-DFlash2 \
MARLIN_DIR=/models/GLM-5.3-Flash-tr3-4bpw-marlin-int4-gs64 \
TORCH_EXTENSIONS_DIR=/models/.torch_extensions \
CUDA_VISIBLE_DEVICES=0,2,4,6 \
PROFILE=multimodal \
PORT=3000 \
scripts/serve_glm53_sm80.sh
```

自适应 prefill 已是脚本默认值。要显式固定当前正式参数，可增加：

```bash
ADAPTIVE_PREFILL=1 \
ADAPTIVE_PREFILL_MAX_TOKENS=2048 \
ADAPTIVE_PREFILL_BUSY_TOKENS=1550 \
GPU_MEMORY_UTILIZATION=0.970 \
scripts/serve_glm53_sm80.sh
```

脚本会为 DFlash k=2 自动把 runner 输入容量设为 2050；只有唯一 prompt prefill
使用 2048，出现第二个请求或 decode 时每个长 prefill 块恢复
`LONG_PREFILL_TOKEN_THRESHOLD=256`。繁忙总预算 1550 可同步容纳六路
`6×(256+2 DFlash slots)=1548`，避免 1024 预算将六路拆成 4/2 微批；2050 只是静态
buffer 容量。正式 `6×128K→512` 已验证该配置无 Waiting/Deferred/Preemption；
换用不同并发、上下文或显存规格时仍应重新执行容量门禁。

AutoRound 快照完成校验后，可用 `scripts/run_autoround_gate.py --execute` 执行
自动 A/B：脚本只管理端口 3000 和 GPU `0,2,4,6`，先跑配对 EXL3 基线，再按
1K→8K/32K→128K→500K needle/OCR 的顺序晋级；任何吞吐、容量、调度、JIT 或
正确性失败都会停止候选并恢复正式 EXL3。它会锁定并持续核对端口 3001/GPU
`1,3,5,7` 的 DeepSeek 身份，检测到变化即 fail-closed。

脚本以前台进程运行并监听 `0.0.0.0:3000`。首次启动会加载约 315 GiB 的目标
权重加 sidecar 数据、初始化四个 PP worker、分配 KV cache 并完成 JIT/CUDA
Graph warmup，通常需要数分钟。看到 `Application startup complete` 后再发请求。

三个 profile 的区别：

| `PROFILE` | 视觉 | PP 分层 | cohort barrier | 用途 |
|---|:---:|---|:---:|---|
| `multimodal` | 开启 | `13,12,11,9` | 关闭 | 默认生产服务；当前修正共同窗口 230.24 tok/s |
| `text` | 关闭 | `13,11,11,10` | 关闭 | 普通文本服务，释放视觉激活预算 |
| `text-benchmark` | 关闭 | `13,11,11,10` | **开启** | 仅用于严格同步的 6×128K 基准 |

`text-benchmark` 会等待六个大 prompt 形成 cohort。不要把它用于普通单请求服务，
否则单个请求可能长时间停留在 Waiting。生产 profile 默认 barrier=0。

`--max-model-len=524288` 表示单请求允许的长度上限，不代表 KV cache 能同时驻留
六个 512K 请求。默认容量和调度目标是六路 128K。

## API 验证

同机验证：

```bash
curl http://127.0.0.1:3000/v1/models

curl http://127.0.0.1:3000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5.3-Flash-tr3-4bpw",
    "messages": [{"role": "user", "content": "你是什么模型？"}],
    "max_tokens": 256,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

内网客户端使用 `http://<服务器IP>:3000/v1`。如果连接超时，依次确认：

```bash
ss -ltnp | grep 3000
curl http://127.0.0.1:3000/health
curl http://<服务器IP>:3000/v1/models
```

并检查主机防火墙/安全组是否允许 TCP 3000。将接口暴露到不可信网络前，请在
反向代理层增加鉴权和访问控制；启动脚本本身不配置 API key。

## 运行注意事项

- 首次命中新形状时仍可能触发 Triton JIT，造成明显的单请求延迟尖峰。正式压测
  前先用相同并发和近似长度 warm up。
- 六路性能测试必须检查六个 TTFT、每路 decode TPS、服务日志中的
  `Running/Waiting`，不能只看一个聚合数字。
- DFlash2 `k=2` 是当前 SM80/PP4 的正式选择。更大的 k 虽可能提高接受长度，
  也会增加 draft、KV 和 verify 成本；本机实测没有稳定超过 k=2。
- 当前 EXL3 多模态 profile 的 `gpu_memory_utilization=0.970` 已通过六路 128K
  容量门禁；0.963/0.965 都发生过一次 hybrid KV 抢占。该值不能直接复用到不同
  checkpoint 或 PP 分层，换模型后必须从较低值重新 profile。
- CPU 高占用主要来自 checkpoint 读取/反序列化、tokenization、PP worker IPC、
  scheduler 和长 prompt prefill 的 host-side 准备。加载期 GPU 利用率低是正常的。
- 服务使用 TP1/PP4。当前 EXL3 reader 明确拒绝 TP>1；不要直接改成 TP4。
- 多模态 profile 不添加 `--language-model-only` 或 `--skip-mm-profiling`，这样
  vLLM 才会为视觉激活正确预留显存。
- 停止服务请向前台进程发送 `SIGTERM`/`Ctrl-C`。若交给 systemd、supervisord
  或容器运行，应让进程管理器直接管理该脚本，不要再套一层后台 `nohup`。

## 开发与测试

核心快速测试示例：

```bash
.venv/bin/python -m pytest -q \
  tests/quantization/test_exl3.py \
  tests/v1/core/test_async_scheduler.py \
  tests/v1/core/test_kv_cache_utils.py \
  tests/v1/worker/test_gpu_block_table.py \
  tests/v1/worker/test_gpu_worker.py
```

本仓库还包含 EXL3/Marlin microbench、SM80 attention tests、PP 通信测试和 KV
cache 单元测试。GPU kernel 测试依赖 SM80/CUDA 环境，不适合在普通 CPU CI 中
全量运行。

## 目录

| 路径 | 内容 |
|---|---|
| `vllm/model_executor/layers/quantization/exl3.py` | EXL3 reader、Marlin sidecar 和 routed-MoE 执行路径 |
| `vllm/v1/attention/` | SM80 sparse MLA、kpool、DFlash 与 attention 修改 |
| `vllm/v1/core/sched/` | PP phase/pairpack/cohort 调度修改 |
| `scripts/serve_glm53_sm80.sh` | 可移植的三 profile 启动脚本 |
| `scripts/build_exllamav3_ext.py` | stock/experimental EXL3 native extension builder |
| `scripts/convert_exl3_to_marlin.py` | 逐层生成 Marlin INT4 sidecar |
| `chat_templates/` | 支持视觉、工具调用和 thinking switch 的模板 |
| `benchmarks/kernels/` | EXL3/Marlin microbench 与 sidecar 验证 |
| `third_party/exllamav3_ext_shared_had/` | 实验性 ExLlamaV3 native extension 修改快照 |

## License 与来源

vLLM 代码使用 Apache-2.0，见 [`LICENSE`](LICENSE)。EXL3 native extension
来源于 ExLlamaV3 0.0.43，使用 MIT License；详细来源与许可见
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。本仓库不包含模型权重、
Marlin sidecar 或 DFlash2 checkpoint。

感谢 [vLLM](https://github.com/vllm-project/vllm)、
[ExLlamaV3](https://github.com/turboderp-org/exllamav3)、
[DFlash](https://github.com/z-lab/dflash)、Z.ai、Inco AI 以及 GLM-5.3 社区
相关工作的作者。本项目不隶属于上述组织。

## English summary

This repository is an unofficial vLLM fork optimized for serving the selective
EXL3/MCG GLM-5.3-Flash checkpoint on four NVIDIA CMP 170HX (SM80) GPUs. It
combines PP4 phase-aware scheduling, sparse MLA kernels for SM80, offline
Marlin INT4 routed-expert sidecars, DFlash2 speculative decoding, a 512K
request limit, and multimodal OpenAI-compatible serving.

The current multimodal production profile measured 230.24 aggregate decode
tok/s for six synchronized 128K prompts with 512 output tokens, using the
corrected common decode window and no waiting, deferral, or preemption. The
older text-only 222.48 tok/s result used a last-TTFT/hot-run window and is not a
strictly paired comparison. These are decode-only aggregate measurements on
the qualified four-GPU host, not end-to-end rates or universal performance
claims. Follow the Chinese deployment guide above for the exact profiles,
dependencies, conversion process, licenses, and operational caveats.
