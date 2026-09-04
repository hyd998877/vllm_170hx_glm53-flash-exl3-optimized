# GLM-5.3 并发与速度优化：快速验证执行方案

状态：**资深二次复核 APPROVE；Batch 1 已于 2026-09-02 执行完成（无候选晋级）**
计划基线：commit `5bd3f5f8`，GPU 0–3，端口 30002  
适用硬件：4× CMP 170HX / SM80；不修改 GPU 功耗、时钟或硬件拓扑

本计划在 2026-09-01 经资深推理系统/性能工程师复核后修订。修订重点是：未实现
的热切换默认关闭、补充 128K 短探针、提高重复测量要求，以及对进程和 GPU 的
fail-closed 保护。

本文的目的不是把所有候选直接跑一遍完整的 `6×128K`，而是在不牺牲正确性和结论可信度的前提下，用逐级门禁快速淘汰无效方案。只有通过便宜代理测试的候选，才进入昂贵的 128K 正式测试。

本版特别采用**低介入批处理模式**：Codex 不参与每个请求、每个日志片段或每个
候选的中途决策。批准后由本地 runner 按预先冻结的 manifest 自动执行、判定和
恢复；Codex 只在批次开始前审查一次，在批次结束后复核一次。这样可以避免等待
智能体响应成为测试链路的一部分。

Batch 1 收尾记录：三个 PP4 分层候选均在 L3 门禁淘汰，未改变正式配置。首次
恢复 formal 时发现 runner 将 formal 配置误当作候选访问 `id`，在服务已经健康
监听后产生误报 `ABORT_REVIEW`。提交 `db63d808` 按 `child_role` 区分 formal/
candidate，并加入回归测试；资深 reviewer 二次批准 recovery-only 收尾。formal
已只读核验通过（`0.0.0.0:30002`、`/health=200`、512K max context），原始
`ABORT_REVIEW` 保留作审计，详细结果见批次目录的 `RECOVERY_VERIFIED.json`。

## 0. 智能体参与边界（低介入模式）

| 时点 | Codex/用户做什么 | 测试期间谁负责 |
|---|---|---|
| 批次开始前（一次） | 审查候选、门槛、GPU/PID 范围、回滚命令；用户批准批次 | 不再临时改参数 |
| 批次执行中 | 不参与；只在 runner 明确触发 `ABORT_REVIEW` 时介入 | runner、HTTP 客户端、日志/指标采集器 |
| 批次结束后（一次） | 审核汇总、确认冠军和下一批；决定是否发布 | runner 已完成清理和正式服务恢复 |

`ABORT_REVIEW` 只允许以下情况触发：检测到未知 PID/端口占用、GPU4–7 业务状态
变化、数据格式损坏、连续两次结果波动超过 5%、或出现未列入策略的错误。OOM、
HTTP 超时、Waiting/Deferred、JIT 警告和候选性能回退均按预设规则自动记录并跳过，
不等待智能体解释。

审查确定的 P0/P1 修订：

- P0：仓库当前没有 R1 Unix-socket 控制面；`k`、MBT、CUDA Graph 形状等也并非
  通用热切换参数。R1 在实现并通过 sentinel 等价性测试前一律禁用，所有实际
  候选按 R2 干净重启执行，不能把“计划中的接口”当作已有功能。
- P0：自动清理前必须拿 runner 独占锁，并使用全新 output_dir；核对 PID 的 `/proc`
  starttime、cmdline、cwd、环境 allowlist、进程组、实际监听地址和 GPU UUID/PID。
  身份不匹配时只触发 `ABORT_REVIEW`，绝不发送 TERM/KILL。GPU4–7 的 PID 快照
  在整批期间必须不变。
- P0：正式服务停止前要求 running/waiting/deferred 连续 5 秒为 0；候选父进程
  用 `wait()` reap 后再检查进程组，残留 worker 直接中止且禁止恢复冲突服务。
- P0：runner 由独立 `fast_opt_watchdog.py` 启动和监护；runner 的普通异常、
  SIGTERM/SIGHUP 都进入恢复路径，未知 listener、残留 worker 或 GPU 状态变化
  时 watchdog 只写 `ABORT_REVIEW`，不猜测性杀进程。
- P1：两次样本和 3% 差异不能形成可靠结论。性能门槛改为配对 seed 的至少三次
  计时样本；近门槛候选用 bootstrap 95% CI，CI 跨越淘汰线时自动标记
  `inconclusive`，最多重测一次。
- P0：decode 聚合使用 `max(last)-min(first)` 的共同时间窗，避免按最后一路 TTFT
  截断造成虚高；运行期间每秒采样并记录 Waiting/Deferred 峰值。
- P0：L3 只是速度/容量初筛；L4 的 32K 与 128K 短探针必须有同口径 paired baseline，
  同时比较 TTFT、decode、排队峰值和 KV tokens 后才算 promoted。
- P1：1K/32K 代理不能证明 128K MLA/KV 行为。涉及 PP、prefill、KV、page 或
  调度的候选，L4 后最多保留两个，必须增加 `6×128K→16/32` 短探针；只有短探针
  通过才允许进入昂贵的 128K→512 正式测试。

### 0.1 runner 的安全状态机

```text
PRECHECK → SNAPSHOT → START_CANDIDATE → HEALTHY
    → L2 → L3(repeat) → L4(optional) → DECIDE
    → NEXT_CANDIDATE → RESTORE_FORMAL → BATCH_REPORT
```

任何节点失败都进入 `CLEANUP`，只清理 manifest 中记录的候选 PID 和临时目录；
不会使用 `killall`、不会杀掉未知 vLLM 进程，也不会自动停止 GPU4–7 上的服务。
`RESTORE_FORMAL` 必须通过 `/health`、端口、模型名和 GPU PID 四项检查后，runner
才退出成功。恢复失败直接 `ABORT_REVIEW`，不继续后续批次。

## 1. 目标、边界和当前基线

优化目标分为三类，不能用一个吞吐数字混在一起：

1. **冷长输入延迟**：降低 128K prompt 的 TTFT 和端到端延迟。
2. **稳态解码速度**：提高单路 decode TPS 和六路聚合 decode TPS。
3. **并发接纳能力**：提高可同时运行的长请求数，且不以 Waiting、抢占或明显降低单路速度作为代价。

当前正式多模态基线：

| 项目 | 基线 |
|---|---:|
| 拓扑 | PP4 / TP1，分层 `13,12,11,9` |
| DFlash | DFlash2，BF16，`k=2` |
| 调度 | MBT=1024，max-num-seqs=6，pairpack，动态 hand-off |
| KV | GMU=0.96，1,184,385 tokens；PP raw blocks `768/375/684/1379` |
| `6×128K→512` TTFT | 约 315.93–316.85 s |
| `6×128K→512` 聚合 decode | 208.01 tok/s |
| 每路 decode | 均值 36.20，中位 36.62，最低 34.46 tok/s |
| 调度状态 | Waiting=0，Deferred=0 |

历史文本专用热测最佳 222.48 tok/s 不是多模态同口径基线。任何候选都必须和相同 profile、相同 prompt 类型、相同并发和 sampling 的基线比较。

当前一次服务冷启动约需 5.5 分钟，其中四个 PP rank 加载模型约 213–230 秒。因此本计划同时控制“正式长测次数”和“重启次数”。

## 2. 核心方法：测试金字塔 + 连续淘汰

每个候选最多经过六级。失败后立即停止，不进入更昂贵的级别。

| 级别 | 测什么 | 典型耗时 | 作用 |
|---|---|---:|---|
| L0 静态检查 | 配置、shape、分片、显存账本、定向单测 | 1–10 min | 不加载完整服务即可排除不支持方案 |
| L1 kernel/组件 | 数值正确性、CUDA event 微基准、通信带宽 | 1–5 min | 判断局部优化是否有足够大的理论收益 |
| L2 服务 smoke | `1×1K→16`，文本；涉及视觉时再做一次 OCR | <1 min | 验证启动、API、正确性和无崩溃 |
| L3 decode 筛选 | `6×1K→512`，预热 1 次、计时 3 次 | 1–2 min | 快速判断 decode、PP 和 CPU 控制路径 |
| L4 上下文趋势 | `6×8K→64` + `6×32K→64`；晋级者再跑 `6×32K→512` | 2–5 min | 检查 prefill、KV、长上下文 attention 趋势 |
| L5 正式验收 | `6×128K→512` 至少三次冷且 prompt seed 不同；最终冠军再跑 `→8192` | 每候选 18–40 min | 形成可发布结论 |

这里的“冷且 seed 不同”指每次请求从首个用户 token 起都不同，禁止后续测试复用之前的 prefix cache。测试工具使用 `--prompt-seed`，并在结果中记录服务端 prefix-cache hit delta；否则热 cache 会污染 TTFT。

### 2.1 为什么代理测试合理

- `6×1K→512` 保留了六路 decode batch、DFlash、LM head、PP hand-off、CPU scheduler 和 CUDA Graph，适合快速筛选稳态解码优化。
- `8K/32K` 比 128K 快得多，但能暴露 prefill slope、KV page 对齐、稀疏 MLA、PP stage 失衡以及随着上下文增长出现的退化。
- kernel 微基准只用于证明局部改动有潜力，不能代替端到端结果。
- `128K` 只用于验证代理结果能否外推，并作为最终正式数字。

### 2.2 统一晋级门槛

所有性能比较先完成一次不计时 warmup。L3/L4 至少使用三个配对 seed 的计时样本；
明显远离门槛时可在三次后结束，接近门槛时用 bootstrap 95% CI 决定是否需要一次
额外复测。L5 正式结果至少三个独立冷测；两次结果只可标为 `functional-only`，
不得写成性能通过。

**正确性硬门槛：**

- 所有请求完成，completion token 数符合要求；无 HTTP 失败、engine dead、OOM、NaN、非法访存或 worker 重启。
- 不改变数值路径的调度、缓存和通信改动，在 temperature=0 时必须和对应基线做 token 级输出比较。TP 归约顺序、KV dtype 等会合理改变浮点舍入的候选，若 token 不同则继续做中间 tensor 容差、needle/任务正确性和至少 64-token 的逐步对照；不能仅凭文本看起来合理就通过。
- 多模态相关改动必须通过固定图片 OCR smoke。
- 所有标为 cold 的测试，服务端 prefix-cache hit delta 必须为 0；命中非零时该
  样本作废，由 runner 自动更换 seed 重跑一次，不能把热 cache 结果纳入比较。
- 不得出现 Waiting/Deferred/抢占，除非该实验专门测试接纳上限；出现时不得把错峰吞吐当作同步吞吐。

**速度候选晋级门槛：**

- L1 局部 kernel 至少快 10%，且按 Amdahl 估算端到端潜在收益至少 2%；或该改动消除已确认的同步/CPU 阻塞。
- L3 聚合 decode 中位数提高至少 5%，且配对 bootstrap 95% CI 下界仍高于 3%；最低单路不得下降超过 3%。
- L4 `32K` 聚合 decode 至少提高 3%，TTFT 不得下降超过 3%，KV 容量不得下降超过 5%。
- L5 三次独立冷测的聚合 decode 中位数至少提高 3%，bootstrap 95% CI 下界高于 0%，每路中位和最低值均不得回退；TTFT、E2E、Waiting 和 KV 同时单独报告。

**容量候选晋级门槛：**

- 瓶颈 PP rank 的 raw KV blocks 或总 KV tokens 至少提高 15%；
- 六路速度回退不超过 3%，且正式六路无 Waiting/Deferred；
- 只有容量账本支持时才尝试 8 路。8 路测试需同时报告聚合值和每路值，不能只看聚合吞吐。

低于门槛的差异记为噪声或失败，不继续 128K。若局部收益很大而服务级结果刚好低于门槛，仅允许一次“交互救援测试”，避免无限组合搜索。

### 2.3 自适应重复次数和按机制选 workload

不再对每个候选机械地跑相同次数，但任何性能结论至少有三个配对样本：

- L3 先跑三次。若三次相对基线差异大于 8% 且 bootstrap CI 不跨 5% 门槛，可直接晋级或淘汰；接近门槛时只允许自动追加一次。
- decode-only 候选不重复跑两个短 prefill workload：L3 后直接跑 `6×32K→512`。
- prefill/KV 候选不浪费长输出：跑 `6×8K/32K→64`，只有晋级才补 512 输出。
- prefix-cache 候选使用冷/热配对，不纳入独立冷请求速度排名。
- 容量候选先按 ledger 淘汰，再做 8 路；理论预算不足时不发送注定排队的请求。

基线在每三个候选后插入一次短复测。若前后基线漂移超过 3%，runner 自动把该区间标为 `inconclusive` 并仅重跑该区间；不请求 Codex 临时判断。

## 3. 避免组合爆炸的实验设计

不做 PP 分层 × TP/PP × MBT × max-seqs × DFlash k 的全因子排列。采用 `champion/challenger` 连续搜索：

1. 每一轮只有一个当前冠军。
2. 候选只和同口径冠军比较。
3. 通过 L3/L4 才替换冠军。
4. 下一类参数只在新的冠军上搜索。
5. 最后只组合已经独立获胜且机制不冲突的优化。

这样会牺牲穷举所有高阶交互，但能把几十到上百个长测压缩为 2–4 个正式 128K 候选。发现强交互证据时，再增加一组定向组合，而不是展开整个笛卡尔积。

### 3.1 按“是否需要重载模型”分组

为了进一步减少 5.5 分钟的加载成本，候选分为三类：

| 类别 | 例子 | 执行方式 |
|---|---|---|
| R0 请求级 | prompt 长度、并发、prose/code、prefix 冷/热 | 同一服务连续运行 |
| R1（当前禁用） | 计划中的 k policy、MBT 配额、kernel dispatch、计时开关 | 仓库尚无实现；不得在本轮使用。只有实现受控 Unix socket、sentinel 等价性测试和回滚后，下一轮才可启用 |
| R2 必须重启 | PP/TP、PP 分层、KV dtype/page、CUDA Graph 内存形状、权重布局 | 一个配置只加载一次，加载后连续跑完其全部 L2–L4 |

本轮全部按 R0/R2 执行。未来若实现 R1，控制面只存在于实验分支，不开放 TCP，
不允许修改模型路径、GPU、端口、显存比例或任意环境变量；先对一个 sentinel 候选
比较运行时切换和干净重启，差异超过 2% 就禁用 R1。正式 128K 验收始终使用干净
重启的单一冠军配置，避免状态污染。

源码 kernel 候选可在同一实验二进制中保留 baseline/candidate 两条 allowlist dispatch 路径，先在一次加载中筛选；只有胜出实现才 cherry-pick 到干净 release-candidate 分支并正式重启。这是测试工具，不改变最终部署路径。

### 3.2 冻结 manifest，runner 自主选择冠军

每批开始前生成不可变 manifest，记录候选依赖、晋级门槛、最大重启数、超时和正式恢复配置。runner 使用确定性规则执行 successive halving：L3 淘汰下半区，L4 最多保留两个，每类最多一个进入 L5。运行期间不调用模型、Codex 或外部决策服务。

manifest 的 SHA256 写入所有结果。批次开始后不允许 Codex边看结果边增加候选；新想法进入下一份 manifest，从而避免测试过程中不断改变目标。

## 4. 各优化方向的快速验证方式

| ID | 方向 | 最便宜的有效验证 | 晋级后的服务测试 | 停止条件 |
|---:|---|---|---|---|
| A1 | PP4 分层/KV 平衡 | 启动 ledger + 每 rank 权重/峰值显存；候选 `13,11,11,10`、`12,11,11,11`、`12,12,10,11` | L3、L4；只保留最多 2 个进入 L5 | raw blocks 更差且 stage 时间未改善，直接淘汰 |
| A2 | PP2×TP2 / PP1×TP4 | 先做 EXL3/DFlash 分片加载检查和 NCCL all-reduce 微基准 | 先 PP2×TP2 的 L2/L3/L4；PP1×TP4 只有通信预算合理才启动 | 不支持分片、通信预测吃掉计算收益或 L3 回退 >5% |
| A3 | MBT/调度 | 复用已有 1024/1536/2048 结果，只新增粗点 768 | L3/L4；768 胜出后才二分补 640 或 896 | 768 无收益，不再扫描更密网格 |
| A4 | max-num-seqs/KV 接纳 | 用启动 ledger 计算安全 token 预算，不先发 128K | 最佳内存配置上做 `8×32K→128`；通过才做 `8×128K→128` | 理论预算不足 1.15×，不冒险长测 |
| B1 | static PP direct-recv | 已有 8 个定向测试；补 graph padded-row/alias/事件顺序测试 | `6×1K→64` 后立即 `6×1K→512` | 任一 graph/shape 错误，立即关闭环境开关 |
| B2 | DFlash 自适应 k | 用 event 计时和日志离线回放，按 acceptance 比较 k=1/2/3 的每接受 token 成本 | prose/code 各跑 L3；L4 检查长上下文 | 两类文本任一显著回退，保留固定 k=2 |
| B3 | selector/top-k/rejection/采样融合 | 组件数值对照 + CUDA event，典型 rows=6/12/18 | L3，两类文本；再做 L4 | kernel <10% 或端到端预测 <2%，不集成 |
| B4 | CUDA Graph/JIT warmup | 收集真实 specialization key；检查计时区间内无新 JIT 日志 | 冷、热各一次 L3，报告 p50/p95 | 只增加 graph 显存却不降冷抖动，回退 |
| B5 | partial prefix + Mamba state | 2061/8704-token token级正确性回归；命中量必须与 state boundary 一致 | 8K/32K 同前缀冷/热对照；最终 128K 同前缀 | 任何输出不一致或错误命中，立即失败 |
| B6 | prefill/MLA/indexer kernel | event 微基准覆盖 8K/32K chunk shape | `6×8K/32K→64`；晋级候选再做 `6×128K→16/32` 短探针 | 32K TTFT改善 <3% 或 128K 短探针异常，不跑 128K→512 |
| B7 | CPU/EngineCore | eBPF/py-spy 或 wall timer，只读定位 scheduler、tokenizer、copy 占比 | L3 同时采集单核占用、PP event 空洞 | 未定位 ≥5% wall 热点，不进行盲目并行化 |
| B8 | 混合 KV dtype/page | 单层数值误差、backend 能力检查、每 group 字节账本 | `1×8K` needle + `6×32K→128` | backend fallback、正确性失败或速度回退 >3% |
| B9 | SM80 EXL3/INT4 fused kernel | 真实层/真实 rows 的 kernel microbench，覆盖 routed/non-routed | L3/L4 | 局部 <10% 或 Amdahl 预测 <2% |
| B10 | LM head/vocab top-k 融合 | rows=6/12/18、vocab=154880 的正确性和 event microbench | prose/code L3 | top-k/output 不一致或端到端 <3% |

普通 FP8 draft、greedy draft、固定 `k=4+`、GMU=0.97、MBT=1536/2048 已有失败或回退证据，不重复消耗时间。CPU/NVMe KV offload 只扩容量且会增加数据搬运，不作为速度主线。

## 5. 按重启批次执行

一次冷启动约 5.5 分钟，因此把测试按必须重启的配置分批，并为每批设置上限。

### Batch 0：工具和基线校准，不重启

- 为 benchmark 增加 prompt seed、结果 schema 校验和 prefix hit delta。
- 增加自动健康检查、超时、日志错误扫描、Prometheus 快照和 GPU/CPU 采样。
- 增加 manifest 解析、确定性晋级、原子状态文件、独立进程运行和正式服务恢复。
- 在当前正式服务上做 L2 和一组 L3，验证快速代理结果可复现。
- 采集 CUDA event 分段基线：prefill、draft、verify、selector/rejection、PP send/recv。

预算：30–60 分钟开发/校准；服务测试约 5 分钟。

Batch 0 完成后，配置筛选 Batch 1–3 作为一份 manifest 连续无人值守执行；中间不等待 Codex 回复。runner 会把上一轮冠军自动代入下一类候选，最终恢复正式服务并生成一次总报告。

### Batch 1：PP4 分层 successive halving

每个候选只做：启动 ledger → L2 → L3 → `6×32K→64`。失败立即换下一个。最多两个候选跑 `6×32K→512`，最多一个候选进入 128K。

预算：3 个候选、3 次重启，约 25–40 分钟。

### Batch 2：并行拓扑

先离线检查 TP 分片和通信；优先 PP2×TP2。PP1×TP4 只有在 L0/L1 证明合理时才加载完整服务。每种拓扑使用自己的均衡分层，不强行复用 PP4 分层。

预算：1–2 次重启，约 15–30 分钟。启动失败只诊断一次，不在该批无限修复。

### Batch 3：调度和容量

- 在胜出拓扑上只新增 MBT=768；胜出才补一个 640/896 二分点。
- 把 `max-num-seqs=8` 与胜出的 KV/分层配置合并测试，不单独展开所有组合。
- 分别记录六路速度目标和八路容量目标，二者不互相替代。

预算：1–2 次重启，约 15–30 分钟。

### Batch 4：源码功能探针

先加入只在环境变量开启时生效的计时和实验路径。每个功能使用独立 git 分支，以当前冠军 commit 为基点；定向单测和 L1 通过后才启动服务。

执行顺序：

1. static direct-recv graph-safe 修复；
2. adaptive k；
3. selector/rejection/LM-head 融合；
4. partial prefix Mamba state；
5. JIT warmup；
6. 混合 KV 和 SM80 专用 kernel 可行性。

每个功能服务级筛选最多一次重启；未过 L3/L4 就不合并。源码实现时间取决于问题复杂度，和 GPU 测试预算分开记录。

Codex 只负责实现代码和在运行前把候选加入 manifest。测试启动以后不再通过工具逐条发送请求或人工看日志决定下一步；runner 独立完成该功能的 L0–L4，并返回单个 `DONE` 或 `ABORT_REVIEW` 状态。

### Batch 5：胜出组合和正式验收

- 只合并机制兼容且单项已晋级的候选，形成一个 release candidate。
- 跑至少三次不同 prompt seed 的 `6×128K→512`。
- 达标后跑 prose、code 各一次 `6×128K→512`。
- 最终冠军才跑一次 `6×128K→8192`；如果与固定 k=2 差异在噪声内，不再重复。
- 多模态 OCR、512K needle、API smoke 和故障恢复测试作为发布正确性验收。

预算：约 45–75 分钟。

## 6. 时间上限和预期总量

在代码候选已经实现的前提下，GPU 验证预计如下：

| 项目 | 上限 |
|---|---:|
| 配置筛选重启 | 最多 7 次，约 39 分钟纯加载时间 |
| 通过 L0/L1 的源码候选 | 每项最多 1 次，首轮最多 6 次 |
| 最终组合和正式恢复 | 最多 2 次 |
| 快速 L2/L3 | 约 20–35 分钟 |
| 8K/32K L4 | 约 25–45 分钟 |
| 128K/8192 正式验收 | 约 45–75 分钟 |
| 日志整理与恢复正式服务 | 约 20–30 分钟 |
| **GPU 实验总计** | **通常约 3–4.5 小时；6 小时为暂停复盘硬上限** |

这是首轮上限，不包含编写新 kernel、修复 TP 兼容、实现 partial Mamba state 等研发时间。预计 10 次左右重启；只有六个源码方向全部通过便宜门禁时，最坏才接近 15 次。某方向在 L0/L1 失败时，总时间会明显缩短。累计 GPU 实验达到 6 小时仍未完成时，必须停止并提交阶段结果，由用户决定是否扩展预算。

如果 GPU4–7 的现有业务后来明确允许停止，可开第二条 PP4 测试 lane（建议端口 30003）缩短墙钟时间。两条 lane 必须各跑自己的 paired baseline，不能把 GPU0–3 的基线直接用于 GPU4–7；未经明确许可，不停止当前 GPU4–7 上的 DeepSeek 服务。

低介入模式把 Codex 的测试决策次数从“每候选/每级一次”降为“每批两次”。R1 运行时切换若通过 sentinel 等价性验证，预计还可少 2–4 次模型重载。对应的目标墙钟时间为：配置筛选约 60–100 分钟，已实现源码候选的快速筛选每项约 5–15 分钟，最终正式验收约 45–75 分钟。研发编码时间仍单独计算。

## 7. 自动化、结果目录和可复现性

计划执行前增加一个 fail-fast runner。每个实验写入独立目录：

```text
/mnt/nvme0/keys-vllm-glm53/runtime/fast-opt-YYYYMMDD/
  manifest.json          # commit、branch、模型 hash、参数、环境、GPU 拓扑
  manifest.sha256
  STATUS.json            # 原子更新：当前候选、阶段、已用时间、最近心跳
  events.jsonl           # runner 的结构化事件，不依赖 Codex 对话记录
  runner.pid
  baseline/
  A1-p13111110/
    launch.env
    server.log
    metrics-before.prom
    smoke.json
    c6-1k-512-r{1,2,3}.json
    c6-32k-64.json
    gpu-samples.csv
    cpu-samples.csv
    decision.md
  summary.json
  RESULTS.md
  DONE                    # 成功完成；或 ABORT_REVIEW
```

runner 对每个候选执行：

1. 校验 GPU0–3 和端口 30002 的精确 PID；只停止本计划记录的服务 PID，不使用 `killall`。
2. 启动并等待 `/health`，420 秒未健康则保存日志并失败退出。
3. 从日志解析 PP 分层、每 rank 显存、raw KV blocks、最终 KV tokens、graph/JIT。
4. 依次执行 L2/L3/L4；每一级自动计算相对基线和门槛。
5. 扫描 OOM、Waiting、Deferred、engine dead、HTTP failure、运行期 JIT。
6. 写入 `promoted/rejected/inconclusive`，只有 `promoted` 才进入下一层。
7. 实验结束恢复已验证正式参数，并复核文本、视觉和 `/health`。

runner 由独立 watchdog 进程启动，不依附 Codex 的 shell/PTY 生命周期。Codex 启动后即可退出当前交互，不做高频轮询；需要查看状态时只读取一次 `STATUS.json`。runner 每 10 秒原子更新心跳，单个请求和单次启动都有硬超时。watchdog 只在端口空闲、GPU0–3 无残留且 GPU4–7 快照未变化时恢复正式服务；未知状态 fail-closed，禁止把“猜测性恢复”描述成幂等恢复。

确定性自动判定包括：

- `reject_continue`：正确性失败、OOM、超时、Waiting/Deferred、性能低于淘汰线；清理候选并继续。
- `promote_continue`：达到门槛，进入下一层或替换当前冠军。
- `inconclusive_retry_once`：基线漂移或结果落在不确定区间，只自动复测一次。
- `abort_restore`：未知 PID、保护 GPU 被触碰、结果 schema 损坏或正式服务无法恢复；停止整批、先恢复再等待复核。

runner 不修改代码、不创建 git commit、不 push、不自行扩大候选集合，也不会根据日志文本生成新的优化方案。这样测试速度不受智能体推理速度影响，同时把研发判断和机械执行清晰分开。

所有源码实验遵循：

```text
main（已验证）
  └─ exp/<ID>-<short-name>-YYYYMMDD
       └─ 单测 + 快速结果 + 独立 commit
```

失败实现不混入正式分支；合并后的 release candidate 再建立单独分支。禁止在有未提交修改的工作树上自动切换实验分支。

## 8. 最终报告格式

每个候选都按同一表格报告，缺少任一关键字段不得写“通过”：

| 候选 | 正确性 | 启动 | KV/最小 blocks | TTFT | E2E | 聚合 decode | 每路中位/最低 | Waiting/Deferred | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|

结论只有五种：

- `functional-only`：功能正确，尚无性能证据；
- `promoted`：通过代理测试，进入正式测试；
- `passed`：正式同口径测试达到门槛；
- `rejected`：正确但无收益或明显回退；
- `failed`：正确性、稳定性或容量失败。

## 9. 批准后实际开始点

批准本计划后，先执行 Batch 0，只增加 benchmark/runner、计时开关和实验分支，不改变正式算法。runner 的定向单测、dry-run 和一次故障恢复演练通过后，自动开始 Batch 1–3 配置 campaign；中间不需要 Codex 或用户逐项确认。若代理结果本身波动超过 5%，runner 自动复测一次，仍不稳定则恢复正式服务并生成 `ABORT_REVIEW`。

当前实现的最小入口为：

```bash
# 只校验 manifest，不触碰正在运行的 30002
/mnt/nvme0/keys-vllm-glm53/.venv/bin/python scripts/fast_opt_runner.py \
  --manifest runtime/fast-opt-batch1.json

# 资深二次批准后才执行；watchdog 监护 runner 和正式恢复
/mnt/nvme0/keys-vllm-glm53/.venv/bin/python scripts/fast_opt_watchdog.py \
  --manifest runtime/fast-opt-batch1.json --execute
```

`runtime/fast-opt-batch1.json` 目前只包含已支持的 PP4 分层候选；PP2×TP2、TP4、
static direct-recv 和源码 kernel 方向在各自 L0/L1 实现完成前不会伪装成可执行候选。

第一次阶段汇报节点改为 Batch 1–3 整批结束：届时一次性提交基线校准、PP 分层、PP2×TP2/PP1×TP4、MBT、接纳容量的 ledger/L3/L4 结果、自动淘汰原因和当前冠军。Batch 4 每个已实现源码功能同样按“一次启动、一次结果”执行，不在测试中间等待智能体。未经批准，当前 30002 服务保持不变。

## 10. Batch 2–5 执行收尾（2026-09-02）

本轮按上述门禁继续执行，目标服务始终限定为 GPU0–3、PP4/TP1、端口 30002；GPU4–7
的 DeepSeek 进程未停止、未重启、未改变功耗或时钟。Batch 2–4 的候选均已完成
静态或服务级判定，未通过门禁的开关保持关闭。

### Batch 2：拓扑

PP2×TP2 和 PP1×TP4 在 L0 被淘汰：当前 EXL3 MoE 实现要求 TP=1，且 GPU0–3
之间只有 PCIe PIX、没有 NVLink。未发生服务重启，证据为
`runtime/fast-opt-batch2-l0.json`。

### Batch 3：调度

MBT=768 完成三次配对 L3 和长上下文代理。聚合中位相对 MBT=1024 为
`+1.084%`，bootstrap CI95 为 `[-0.933%, +1.972%]`，且首轮出现 Waiting=2，
未达到速度或稳定性门槛；正式配置保持 MBT=1024。证据目录为
`runtime/fast-opt-20260902/batch3-mbt768/`。

### Batch 4：DFlash、PP hand-off 与 kernel 门禁

- static direct-recv：定向单测通过，但两次服务 smoke 分别触发 graph buffer
  重叠拷贝和 padded-row 形状不一致；服务级候选回退约 9%，未启用。
- adaptive k：`DFLASH_BATCH_SCHEDULE_JSON` 已实现为 opt-in 配置并通过 6 项
  fixed-PP worker 测试；`k=3 (batch 1–2) / k=2 (batch 3–6)` 的 L3 中位相对固定
  k=2 为 `-2.920%`，CI95 `[-3.845%, -0.498%]`，未启用。
- prefix/Mamba/kpool/runner 定向门禁：`122 passed, 1 skipped`；DFlash
  selector/causality 逻辑门禁 `18 passed`。GPU-only prepare/rejection 长测因
  GPU0–3 正式服务占用而不强行并行；已有服务级 A/B 已覆盖真实 GPU 路径。
- kernel/selector 审计：selector walk、draft-logit cache 和 rejection 已各自
  使用 Triton kernel；当前瓶颈候选仍包含 154880 词表的 FP32 sampling-param
  materialization。CMP170HX 返回 `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED`，没有
  足够 event 占比证据，不合入未经证明的融合改动。已有 fused context-KV
  projection 微基准在 6/18/256 token 分别为 `1.94x/1.56x/1.26x`，保留现路径。
- JIT 复核：verbose specialization 证据显示剩余 kpool/FP8 MQA 首次编译并非
  完全漏掉 kernel，而是 warmup dummy 只覆盖了部分指针对齐和标量
  divisibility key。补齐对齐/未对齐及 token-count key 是合理的后续候选，但
  需要独立 manifest、重启和冷/热延迟 A/B；本轮不把未经服务级验证的 warmup
  扩张混入正式配置。

### Batch 5：正式验收

建立 `runtime/fast-opt-batch5.json`，采用当前正式服务的 in-place、no-restart
验收，结果目录为
`/mnt/nvme0/keys-vllm-glm53/runtime/fast-opt-20260902/batch5-formal-acceptance/`。
为保证输入类型可复现，benchmark 新增 `--workload prose|code`，不改变服务算法。
该 JSON 的 schema 明确为 `acceptance_result`，`runner_executable=false`；它是
完成态结果清单，不是含 `baseline/candidates` 的可执行 campaign manifest。

| 测试 | 结果 |
|---|---:|
| prose，3× `6×128K→512`，共同 decode 窗口聚合中位 | **165.75 tok/s** |
| prose，3×每路 decode 中位 | **34.57 tok/s** |
| code，`6×128K→512`，共同窗口聚合/每路中位 | **202.28 / 40.68 tok/s** |
| prose，`6×128K→8192`，共同窗口聚合/每路中位 | **185.81 / 36.37 tok/s** |
| 500K needle | **通过**，实际 500000 tokens，marker 精确命中 |
| 多模态 OCR | **通过**，返回 `Hello, AI world!` |
| API smoke / 512K max context | **通过**，`/v1/models` 报 max_model_len=524288，API 返回 `OK` |
| Waiting/Deferred | 所有正式长测结束均为 0 |
| 故障恢复 | Batch 4 watchdog recovery 已验证；Batch 5 复用同一 fail-closed 路径 |

这里的 165.75/202.28/185.81 使用修正后的共同时间窗口（最早首 token 到最晚
末 token）。历史 208.01/222.48 是旧的 last-TTFT 或热测口径，仍保留作历史记录，
不能与本轮新口径直接比较；因此本轮没有虚构新的“冠军”。正式实例最终恢复并核验
为 `0.0.0.0:30002`，命令行仍为 DFlash2 k=2、视觉路径、PP 分层 `13,12,11,9`。

## 11. 100K+ 冷前缀加速：自适应 prefill（2026-09-04）

线上样本已确认 153,789-token 请求总耗时 335.65 秒，其中 prefill 116.27 秒、
decode 218.78 秒、排队约 0 秒。服务没有 Waiting、Deferred 或 preemption；慢点是
新前缀的实际计算。固定 `long_prefill_token_threshold=256` 会把该输入拆成至少
601 个调度片段，放大 PP4 气泡、host 调度、跨 stage 通信和混合 Mamba/MLA 状态
保存成本。

实验分支 `exp/adaptive-prefill-20260904` 增加二态调度：

- 只有一个未完成请求、该请求仍在首次 prompt prefill 且尚无输出 token 时，单块
  上限提高到 2048；
- 一旦第二个请求进入，或存在 decode/恢复生成，下一调度步恢复配置的 256；
- 繁忙状态每路恢复 256；最初总预算恢复 1024，但服务级六路复核证明它会把请求
  拆成持续变化的 4/2 微批并放大 PP 气泡，因此后续候选改为 1550，恰好容纳
  `6×(256+2 DFlash slots)=1548`，仍低于 2050 静态 runner 容量；
- 并发消失后，如果原请求仍在 prompt prefill，可重新进入 2048；
- 开关默认关闭；生产包装脚本显式启用。DFlash k=2 还需要两个输入槽，因此 runner
  容量设为 2050，净 prefill 上限才是 2048；
- 已经提交到 GPU 的大块不能撤销。Async PP 可同时保有多步 in-flight，因此第二个
  请求的最坏额外等待不只一个块；需在服务 A/B 中记录到达时的在途深度与 p95/max
  ITL，这是 TTFT 收益与在线抢占粒度之间的主要代价。

功能门禁已通过：AsyncScheduler、Mamba 对齐与 partial-prefix-cache 定向测试合计
`95 passed`。正式 3000 服务（GPU `0,2,4,6`）已启用该策略：唯一 128K 冷请求
TTFT 从固定 256-token chunk 的 92.741 秒降到 33.433 秒，改善 64.0%；相同请求热
重放 TTFT 1.077 秒。并发注入时日志确认下一步回退到 256。一次
`6×128K→512` 回归被两个额外业务请求污染，出现 Waiting=2，结果只保留为干扰
样本，不能用其 171.32 tok/s 改写正式基线。

无外部业务、暂停 AutoRound 下载后的正式六路复核暴露了第二个调度问题：共同
decode 窗口仅 37.07 tok/s，每路中位 9.02 tok/s，TTFT 分成
220.27/238.81/287.91 秒三批；全程 Running=6、Waiting=0、Preemption=0。根因不是
KV 容量，而是 busy budget=1024 每轮只能容纳 `4×256`，RUNNING 队列又固定按
FCFS 从头扫描；领先的四个 prefill 会持续占满预算，第五、第六路无法及时追平，
并造成 decode/prefill 长时间混跑。候选修复在并发纯 prefill 时按
`num_computed_tokens` 最少者优先，并让领先 RUNNING 请求为落后的 WAITING prefill
让路。它不恢复固定六路 cohort barrier，也不延迟真正单路的首个 2048-token 块。
首次服务级尝试又遇到一条外部业务请求，且发现满 6 路后第 7 路 WAITING 不能参与
公平水位，否则六个已运行请求会全部为暂时不可接纳的请求让步，形成零工作调度。
候选已把公平集合限制为 RUNNING 加当前剩余 sequence slot 可接纳的 WAITING；满载
时第 7 路不参与水位计算。离线自适应定向回归 `8 passed`、完整 AsyncScheduler
回归 `28 passed`；其中新用例
覆盖“先到一路 2048、随后五路到达”并要求最终进度差不超过 256。需重启后的
服务级 A/B 才能晋级正式配置。失败原始结果为
`runtime/adaptive-prefill-20260904/c6-128k-512-clean-20260904.json`。

部署公平调度后的第二次无外部流量六路验证仍失败：共同 decode 窗口 27.43 tok/s，
每路中位 5.38 tok/s，TTFT spread=96.34 秒，总 wall=494.18 秒。它证明只修正
请求水位而继续保持 1024 总预算，会使 4/2 微批形状不断交替并进一步增加 PP4
气泡。下一候选把繁忙总预算提高到 1550，同时保留每路 chunk=256 和首路领先追平；
离线门禁已验证六路一次同步调度 1536 target tokens，并预留 12 个 DFlash slots。
失败原始结果为
`runtime/adaptive-prefill-fairness-20260904/c6-128k-512-seed20260905.json`。

部署 busy budget=1550 后，六路公平性和 decode 吞吐已经恢复，但 hybrid KV 的
瞬时分组峰值比名义 token capacity 更严格。`gpu_memory_utilization=0.963` 提供
372 个瓶颈共享 block；正式 `6×128K→512` 得到共同 decode `202.51 tok/s`、
每路均值 `39.44 tok/s`、TTFT spread `0.20s`，但最后一段 prefill 仍出现
`required=8, available=5` 并使一路 PREEMPTED。提高到 0.965 后瓶颈池为 382
blocks，`6×128K→16` 仍使 `num_preemptions_total` 从 0 增到 1，尽管没有再次打印
allocation warning；因此两档都不能晋级。下一容量候选直接使用既定安全上限
0.970，不再以名义 `GPU KV cache size` 或测试结束时 0% usage 代替峰值验收。
原始结果为 `runtime/adaptive-prefill-fairness-20260904/c6-128k-512-b1550-u963-tokenclock.json`
和 `c6-128k-16-b1550-u965.json`。

将正式配置提升至安全上限 `gpu_memory_utilization=0.970` 后，启动得到瓶颈池
407 blocks、全局 KV cache `1,153,433 tokens`。最终无外部流量的
`6×128K→512` 通过容量与公平门禁：6/6 返回完整 512 token ID，TTFT
`326.419–326.522s`（spread `0.102s`），Waiting/Deferred=0，
`num_preemptions_total=0`，且日志无 `KV allocation failed`/`PREEMPTED`；共同
decode 窗口 `230.24 tok/s`，每路均值 `45.90`、中位 `41.92`、最低 `38.58`。
相对 0.963 的 `202.51 tok/s` 提升 13.7%，正式候选晋级 0.970。原始结果为
`runtime/adaptive-prefill-fairness-20260904/c6-128k-512-b1550-u970-tokenclock.json`。

正式验证按下列顺序执行，每项只晋级不同时改变其他变量：

1. 同一条 128K/154K 唯一冷前缀分别用 256、1024、2048，记录 prefill 时间、
   TTFT、每 stage GPU 利用率和调度步数；2048 相对 1024 改善不足 3% 就保留 1024。
2. 在 2048 prefill 进行中注入短 decode，验证下一步块大小为 256，并记录 decode
   ITL 的 p50/p95/max；p95 回退超过 10% 则降低空闲块到 1024。
3. 部署并发 prefill 公平调度后回归 `6×128K→512`，要求
   Waiting/Deferred/Preemption 都为 0、TTFT spread≤8 秒，聚合 decode 与当前正式
   基线差异不超过 3%。
4. 重复相同请求验证 prefix-cache 热路径没有回退，并确认 500K needle 与 OCR
   正确性仍通过。

除放大空闲块外，冷请求方向按预期收益/成本排序如下：

1. **提高真实 prefix-cache 命中率**：固定 system prompt、tool 顺序、JSON
   序列化和模板版本，不把时间戳、随机 ID 放在公共前缀。它不能加快第一次真正冷
   计算，但对重复会话通常是收益最大的路径。
2. **会话粘性与跨轮 KV 保留**：同一会话固定路由到同一实例；多实例再评估主机或
   NVMe KV connector。它减少重复 prefill，不会缩短全新前缀的第一次计算。
3. **预填充专用队列/P-D 分离**：prefill 实例用大 MBT，decode 实例保持小块和低
   ITL。理论上最能兼顾 TTFT 与在线 decode，但四卡条件下会引入 KV 传输且无法在
   同一时刻各自保有完整 PP4，需先做容量设计。
4. **优化 GPU 拓扑**：PP 相邻 stage 尽量使用全 PIX 的连续 GPU；当前 0,2,4,6
   在 2→4 间跨 PHB。用户指定拓扑不自动更改，只作为独立 A/B 候选。
5. **长 prefill kernel/JIT**：补齐 1024/2048 shape 的 warmup specialization，
   profile GLM sparse-MLA indexer、KPool 与 Mamba 状态 kernel，再只融合占比明确的
   路径。它能缩短真正冷计算，但开发成本高于调度优化。
6. **请求侧约束**：简单问答降低 reasoning 强度并限制 `max_tokens`。这不改变
   TTFT，但可直接减少已观测到的 218.78 秒 decode，从而改善用户看到的总时延。

不把 DFlash 作为冷 prefill 加速项：它主要提高 decode，既有 A/B 显示其 TTFT
约增加 14%。可研究“prefill 不运行 draft、进入 decode 后启用”，但动态切换涉及
KV/runner 状态，必须独立于本次自适应 chunk 验证。

## 12. 方向 2/3/4/6、AutoRound 与新硬件内核（2026-09-04）

本轮只操作端口 3000 和 GPU `0,2,4,6`；端口 3001/GPU `1,3,5,7` 的 DeepSeek
实例不停止、不重启；不修改 GPU 功耗或时钟。

### 12.1 方向 2、3、4、6 的执行结论

| 方向 | 当前动作与证据 | 状态 | 后续触发条件 |
|---|---|---|---|
| 2 会话粘性/KV 保留 | 新增 `scripts/audit_prefix_requests.py`，按生产 tokenizer/template 比较真实公共 token 前缀，报告首个差异并审计时间戳、UUID、tools/JSON 顺序；单实例 3000 没有路由分叉 | **工具通过，路由不适用** | 出现两个以上 GLM 实例时，按稳定 `session_id` 做一致性哈希；不得把 session ID 当成 prefix 相等的替代品 |
| 3 P/D 分离 | 一份 PP4 已占四卡，另一组四卡由 3001 使用；prefill/decode 各自保有完整模型至少还需四卡，且跨 PCIe PHB 传 128K MLA/KDA/DFlash 状态成本高 | **rejected_capacity** | 增加四张空闲卡，或先证明模型副本可在更少卡完整承载，再做 connector/state 传输微基准 |
| 4 GPU 拓扑 | `0–3` 与 `4–7` 各组内全 PIX，组间 PHB；`0,2,4,6` 的 PP 路径只有 `2→4` 一次跨 PHB，已是跨两个 PCIe 域时的最少跨域边数 | **当前约束下最优/不可 A/B** | 只有允许迁移或停止 3001 后，才比较连续 `0,1,2,3` 或 `4,5,6,7`；不能在业务流量中偷换卡 |
| 6 请求约束 | 短问显式 `max_tokens=64` 且 `enable_thinking=false`，实测 TTFT 0.46–0.50 秒、总耗时 0.86–0.91 秒、14 completion tokens | **通过（客户端策略）** | 由调用方按场景设置；不在服务端全局关闭 reasoning，以免损害复杂任务 |

### 12.2 AutoRound W4A16 候选

候选为 `Intel/GLM-5.3-Flash-W4A16-AutoRound`，配置已静态解析为
`auto_round:auto_gptq / INT4 / group_size=128 / symmetric`。它把 routed experts
量化为 W4A16，而 attention、KDA/MLA、sparse indexer、shared experts、视觉、norm
等 679 个显式条目保留为 16-bit。本机 CUDA 路径会使用 INC/GPTQ Marlin，不会
使用 Intel XPU kernel。

官方模型卡只给出四个短任务：BF16 平均 0.8399，INT4 平均 0.8386，绝对下降
0.0013，相对保留 99.84%。这是低质量损失的正面证据，但没有覆盖代码、OCR、
tool calling、128K/512K needle 或与当前 EXL3 4bpw 的直接比较。

性能不能预先判定更快。当前冠军已把 routed experts 转成更细粒度的 group-size 64
Marlin sidecar；AutoRound 的 group-size 128 scale 开销略小，但 checkpoint 总量约
181.5 GB，保留的 BF16 路径更多，可能增加权重带宽、PP stage 显存和降低 KV 容量。
因此预期为“可能持平或稍慢、质量证据较好”，以实测为准。

执行门禁：

1. 下载 49 个文件/36 个 checkpoint shard 后，逐文件核对 ModelScope API 的 Size 与 SHA-256；
   任一不符不启动。
2. 使用 `scripts/serve_glm53_autoround_sm80.sh` 和独立 runtime 目录加载；先检查
   INC/GPTQ Marlin dispatch、各 PP rank 权重/峰值、KV tokens、视觉 profiling、
   完整 kernel warmup 和 API identity。
3. L2：1K→16、OCR、tool/reasoning smoke；L3：6×1K→512 三个配对 seed；L4：
   8K/32K；只有 L3 中位提升至少 5%、L4 不回退才跑 6×128K→512。
4. 质量门禁至少包含固定文本/代码样本、500K needle、OCR 和工具调用；性能与质量
   同时通过才替换 `/mnt/nvme0/start_glm53_3000.sh` 的正式模型，否则自动恢复 EXL3。

2026-09-04 下载完成；49/49 文件的 Size/SHA-256 校验全部通过，索引实际包含
36 个 checkpoint shard。配置静态门禁通过：`INCConfig.from_config` 解析为
`inc / INT4 / gs128 / symmetric / auto_round:auto_gptq`，相关离线测试
`13 passed`。候选在 SM80 上实际选择 `MARLIN` WNA16 MoE backend，视觉、文本和
forced tool-call smoke 均通过；启动得到 KV capacity `1,459,504 tokens`。

2026-09-05 完成第三轮干净的配对 A/B。两边均先做完整 `6×1K→512` 不计分
warmup，再使用相同三个 seed 计分；指标为从最早首 token 到最晚末 token 的共同
decode 窗口：

| 模型 | warmup | 三次 L3 | 中位 | 相对 EXL3 |
|---|---:|---|---:|---:|
| EXL3 4bpw | 179.51 | 189.72 / 192.71 / 195.94 | **192.71** | 基线 |
| AutoRound W4A16 | 172.92 | 177.95 / 184.10 / 179.96 | **179.96** | **-6.62%** |

AutoRound 的晋级线为 EXL3 中位的 105%，即 `202.35 tok/s`；候选低于晋级线
11.06%，因此状态为 **rejected_performance**。若按端到端 wall-clock 口径，候选
三次中位为 171.45、EXL3 为 183.20，结论相同。脚本按设计停止后续 8K/32K、
128K 和 500K 长测，避免在已失败候选上浪费时间，并恢复正式 EXL3。原始记录位于
`/mnt/nvme0/keys-vllm-glm53/runtime/autoround-ab-final-r3/`；`SUMMARY.json` 的拒绝原因是
`L3 median improvement below 5%`。

后续服务级验证已固化为 `scripts/run_autoround_gate.py`：执行前锁定 3001 和
GPU `1,3,5,7` 的 PID/starttime/cmdline/GPU-process 快照，先对当前 EXL3 重跑
同 seed L3/L4 基线，再排空并切换 3000。AutoRound 依次经过 API/OCR、1K warmup、
三次 6×1K→512、8K/32K、KV capacity、6×128K→512、500K needle 和运行期 JIT
门禁；L3 中位未提高 5% 时即停止，不浪费 128K 时间。无论成功、失败或收到
SIGTERM/SIGHUP，`finally` 都停止候选并恢复 `/mnt/nvme0/start_glm53_3000.sh`；
DeepSeek 身份变化或目标 GPU 出现未知进程则 fail-closed。

门禁的运行期保护检查均通过，结束后另做现场核验：3000 已恢复为
`GLM-5.3-Flash-tr3-4bpw`、监听
`0.0.0.0:3000`、`max_model_len=524288`；3001 仍为原 DeepSeek PID 18519，且
GPU `1,3,5,7` 的进程身份未变化。`SUMMARY.json` 只保存性能判定，不单独保存
before/after 进程快照；进程隔离结论来自门禁成功控制流与上述现场核验。正式配置
不切换至 AutoRound。

### 12.3 完整 warmup

启动 warmup 已覆盖 mHC、KPool cache/tail、prefill/decode MQA logits、Mamba
acceptance 和 CUDA Graph。服务级复核一度发现 DFlash `_prepare_dflash_inputs_kernel`
在首个请求 JIT；根因不是指针对齐，而是七档 DFlash dummy warmup 被放在 V1 分支，
当前 V2 model runner 根本没有执行。

现已在 V2 `warmup_kernels` 收尾阶段调用 DFlash speculator 的 serving-buffer
warmup，分别以 126/62/30/14/6/2/1 个 target query（再加 3 个 DFlash query）覆盖
256/128/64/32/16/8/4 七档 BLOCK_SIZE；不增加模型 forward，只编译输入元数据 kernel。
重启后在 JIT monitor 下依次完成 1K→64、8K→16、128K→16，三段均无运行期 JIT，
因此第 7 项正式通过。warmup 主要消除首次延迟尖峰，不宣称提高稳态 tok/s。

### 12.4 FlashMLA、DeepGEMM、SM90/SM100 与 V4 integration

当前 CMP 170HX 的 compute capability 是 8.0（SM80）。仓库中的 FlashMLA sparse
明确只接受 SM90/SM100，DeepGEMM 只在 Hopper/Blackwell 路径启用，因此两者在
本机不是可打开的优化开关；强制选择会失败或回退，不能产生速度收益。

换到 SM90/SM100 后，FlashMLA/FlashInfer sparse MLA、DeepGEMM、TMA/WGMMA 以及
原生 FP8/FP4 Tensor Core 有明确潜力，尤其能替代当前 SM80 Triton indexer/MQA
fallback。但收益还取决于 checkpoint：AutoRound W4A16 仍主要走 Marlin，并不会
因为安装 DeepGEMM 自动变成 FP8/FP4 模型。应为新硬件重新选择 BF16/FP8/NVFP4
权重与 KV 格式后做 A/B。

DeepSeek V4 专用 sparse MLA/indexer 不能直接移植到 GLM：V4 使用不同的
SWA/hash/compressor、top-k=512、nextn=3、576-byte cache/metadata，而 GLM 使用
KDA/Mamba hybrid state、top-k=2048、kpool=4、nextn=1 和 512-wide NoPE MLA。
可复用的只有设计思想（flatten metadata、融合 Q/RoPE/quant、persistent workspace、
调度计划缓存）。若要落地，必须写 GLM 专用 kernel，并先通过 token/state/KV layout
正确性测试和 ≥10% kernel、≥2% 端到端 Amdahl 门禁；本轮不接入 V4 二进制路径。

## 13. MooncakeStore 双实例前缀共享（2026-09-05）

目标是两路相同 GLM 各占连续四卡（`0–3`、`4–7`，均 TP1/PP4），监听 3000/3001，
通过一个本机 Mooncake master 共享已完成的相同 KV 前缀。该阶段不改变功耗、时钟、
模型数值路径或正式单实例性能结论。

### 13.1 冻结设计

- Mooncake `embedded + tcp`，master `127.0.0.1:50051`，指标端口 50052。
- 每个 PP worker 提供 4 GiB global segment 和 512 MiB local buffer；8 worker 的
  CPU 内存上限约 36 GiB，不启用磁盘 offload。
- 两路均为 `MooncakeStoreConnector / kv_both`，共享同一个带版本的
  `cache_prefix`；固定 `PYTHONHASHSEED=20260905` 和 `sha256`。
- 模型、served-model name、PP 分层 `13,12,11,9`、block size 256、KV auto、
  DFlash2 k=2、multimodal profile 完全一致。
- 混合 Mamba cache 固定为 align；GLM KPool tail 是 request-private scratch，继续
  排除在 transfer groups 外。DFlash/EAGLE 的共享规则由已有 HMA coordinator
  门禁覆盖。
- 两路 lookup IPC 分配不同 `lookup_rpc_port`，且使用短的独立 Unix socket 目录；
  Mooncake 模式关闭 `expandable_segments`，防止已注册 KV 虚拟地址被重映射。

### 13.2 快速门禁与晋级顺序

| 项目 | 验收条件 | 状态 |
|---|---|---|
| Mooncake/CUDA ABI | 模块、master 可加载，版本为 CUDA 13 build | **通过** |
| HMA 定向回归 | worker + HMA E2E + prepare-values 全部通过；最终离线重跑六个 Mooncake 文件 | **146 passed；完整六文件 241 passed** |
| 扩大回归 | Mooncake/HMA/core 扩大集合 | **293 passed；另 34 个在断言前因离线缺少 Hugging Face fixture checkpoint 未运行到功能验证** |
| 资源账本 | 251 GiB RAM、`/dev/shm` 126 GiB、两组 GPU 组内 PIX | **通过** |
| 安全启停 | 独立 PID/log/runtime，端口/GPU/PID fail-closed | **通过** |
| 3000/3001 API | 两路 `/health`、model identity、512K 上限 | **通过** |
| 文本/OCR/tool | 两路精确 `OK`、`Hello, AI world!`、forced `get_weather(Paris)` | **通过** |
| A→B 8K | 8,705 tokens；命中 4,352；输出一致；0 failed keys | **通过；6.642 → 5.448 s** |
| B→A 8K | 8,705 tokens；命中 4,352；输出一致；0 failed keys | **通过；3.564 → 2.380 s** |
| 32K 双向 | 32,769 tokens；每向命中 26,112；0 failed keys | **通过；1.72× / 2.03×** |
| 128K 双向 | 131,073 tokens；每向命中 126,208；输出一致 | **通过；11.01× / 10.80×** |

启动时暴露出的两个部署问题已纳入脚本保护：Mooncake 与
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 不兼容；长 runtime 路径又会超过
Linux `sockaddr_un.sun_path` 的 107 字节上限。前者改用 native allocator，后者改用
每实例短路径。两次都发生在请求验证前，不能计为共享功能通过。

启停入口共用同一把 `flock` 控制锁，并以 `/proc` starttime、PGID、SID、cmdline
联合校验 PID 账本，防止并发启停、PID 复用或伪造 stale PID 导致误杀。下次干净
启动后 master 的 RPC 和 metrics 仅监听 loopback；对外只开放 3000/3001 API。

另有一个不属于本实验的 DeepSeek JIT audit 在 GLM worker 初始化窗口延迟占用
GPU0–3，导致第一次 3000 启动因可用显存不足被 vLLM 拒绝。脚本已增加对“尚未占用
显存、但环境声明会使用目标 GPU”的重叠 vLLM 预检；未知进程只报告，不终止。

服务级验证严格按 `8,705 → 32,769 → 131,073` 晋级，即 8K/32K/128K 目标各加
1 token。Mooncake 只保存其中完整的 4,352-token transfer chunks，余数作为本地
重算后缀，确保能同时验证外部命中与后缀衔接。
每个长度先用唯一前缀测真冷 producer，再等待“完整 chunk 数 × PP4”的
`save_exists`/`save_put` 任务全部排空，最后在 peer 请求相同 token IDs；反向
使用另一唯一前缀。DFlash 需重放一个 4,352-token target block 来重建未传输的
draft KV，所以 8,705-token 用例的正确最小外部命中是 4,352，不是 7,936。

### 13.3 长上下文故障根因与修复

早期实现把每个 HMA cache group 的 Mooncake object 都指向该 PP rank 的全部 KV
backing regions。这在短请求中会被“各 group 恰好使用相同 block ID”掩盖；
长上下文中各 group 的 block table 分化后，某个 group 的 GET 会覆盖其他 group
对应 region，造成 128K 输出错误。修复后每个 transfer group 只登记并传输自己
的 layer regions，分开记录物理 `block_stride` 和真实 `copy_size`，底层 backing
allocation 仍只向 Mooncake 注册一次。

region 顺序不依赖裸 CUDA 地址，而按稳定的 backing-storage 发现顺序和 storage 内
offset 排列；只合并同一 allocation 内真正相邻、stride 相同且合并后不超过物理
block stride 的 region。当前 PP rank 没有 layer 的 projected 空 group 直接标记为
不参与传输。DFlash draft KV 和 KPool tail 均保持 request-private，不进入外部 group。

另外，完成 GET 后必须在实际 KV tensor 所在 GPU 做可见性同步，然后才向
scheduler 发布 load 完成；partial GET 和 GET 异常后的潜在部分写入也必须先 fence。
该 `load_sync` 在 128K 热态中每个 PP rank 平均约
0.1–0.3 ms，相对 3.3 s 跨实例响应可忽略，不应删除。验证器也从“看到
首个 PUT 就开始 peer load”修正为等待全部 PP save 任务排空，避免只命中
部分已写完前缀的竞态。若 GET 或 GPU fence 抛出异常，整次请求涉及的 block IDs
全部失效，不能只失效最后一个 disk-split 子批，否则早先直接写入的 KV 也可能被
scheduler 错误复用。

EngineCore 的 invalid-block 恢复也已扩展到 HMA：BlockPool 的 ID 在同一 engine 内
跨 group 唯一，因此 scheduler 可把错误 ID 映射回 group 的 token 位置，向下取整到
所有 group 共用的 LCM scheduling 边界，并驱逐每个可缓存 group 的依赖后缀。映射
使用 cache manager 的 effective block size，覆盖 DCP 对 attention block 的放大。
错误 ID 只会在 device fence 完成后原子发布，避免主线程提前 free/reuse DMA 尚在写的
block；阻塞 fence 并发门禁与完整 KV-load failure 套件均已覆盖（19 passed）。

### 13.4 正式结果

| prompt | 外部命中 | A→B 冷/共享 | B→A 冷/共享 | 正确性 |
|---:|---:|---:|---:|---|
| 8,705 | 4,352 | 6.642 / 5.448 s | 3.564 / 2.380 s | 首 token 和完整输出一致 |
| 32,769 | 26,112 | 16.793 / 9.783 s | 9.376 / 4.611 s | 首 token 和完整输出一致 |
| 131,073 | 126,208 | 35.920 / 3.262 s | 35.687 / 3.304 s | 首 token 和完整输出一致 |

128K 每个方向均有 600 save keys、132 load keys、0 failed keys，A→B 加速
11.01×，B→A 加速 10.80×。group 隔离后每方向的保存量由旧实现的
9.596 GB 降至 6.047 GB，加载量由 2.111 GB 降至 1.618 GB。正式 JSON 证据位于
`runtime/mooncake-dual-pp4/isolated-final-8705.json`、`isolated-32769.json` 和
`isolated-formal-131073.json`。验证后 master 使用约 21.77/34.36 GB，失败计数为 0。
严格验证器的补充 8,705-token 结果位于 `strict-final-8705.json`，A→B 为
3.199 → 2.418 s、B→A 为 3.186 → 2.413 s，`passed=true`。
这些是重用已存前缀的端到端耗时，不代表新前缀的 prefill 吞吐提高。

上述容错补丁全部加载并干净重启后的最终复验同样通过：热态 8,705-token
A→B 为 3.171 → 2.383 s（1.33×），B→A 为 3.149 → 2.358 s（1.34×）；
131,073-token A→B 为 35.755 → 3.511 s（10.18×），B→A 为
35.441 → 3.283 s（10.80×）。128K 每方向仍精确命中 126,208 tokens、加载
132 keys、0 failed keys，完整 8-token 输出和首 token 一致。证据为
`strict-warm-post-fix-8705.json` 与 `strict-post-fix-131073.json`。
