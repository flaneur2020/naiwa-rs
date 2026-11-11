# M42 — 近期优化与五轮性能验证总结

**Status：** 部分机制验证完成；整体归因、正确性与默认晋升仍在进行。所有新行为保持 opt-in。

**Parent：** [M42 工作记录](m42-token-thread-sync-reads.md) §S13–S32；[M41](m41-cpu-submit-and-resident-dispatch.md)。本篇是 milestone 决策摘要，不替代原始逐轮证据。

## 1. 当前结论

1. CPU-before-DMA、减少 lookahead replacement、单序列 GPU sampling 均有独立 clean 配对收益。
2. 当前组合加入 physical5 CPU pool 后，在数据库 prompt 的五轮同 session 对照中达到 **36.805 ms/token**；Strata suffix drafting 为 **37.794 ms/token**。naiwa 赢4/5轮，但波动更大，不能宣称稳定胜出。
3. 这是 **naiwa 神经 MTP=2 对 Strata suffix/prompt-lookup drafting**，不是双方神经 MTP 对比。Strata 神经 draft pack 仍缺失。
4. 没有把历史无 drafting 的约10ms差距精确分摊。已有诊断能定位结构候选，但不同边界、探针扰动和执行工作变化不能拼成因果饼图。

## 2. 测量口径

- 除最终比较为5次重启外，每臂至少3次重启；两臂AB/BA/AB，三臂轮换顺序。
- 每进程3个请求，每请求128 completion tokens；TPOT分母来自服务端 `usage.completion_tokens - 1 = 127`，不以SSE chunks代替。
- 每进程请求TPOT中位数是一个样本，再统计重启间中位数；重复请求不当成独立重启。
- clean与diagnostic分开；完整文本按对应请求位置比较。首请求与prefix复用请求可能不同。
- 不跨session相减绝对耗时，不把独立机制收益相加。SSE p99不是decode/verify-step p99。
- 默认Rayon/cache2.5GiB/frac0.4/waves1不变；本文“当前组合”不表示已经采用为默认。

## 3. 保留的优化候选

|机制|证据与收益|边界/决策|
|---|---|---|
|`NAIWA_CPU_BEFORE_DMA=1`|固定staged选择后、DMA排队前提交complementary CPU集合。两个repeatable无drafting prompt改善中位数0.907/1.258ms/token；默认Rayon无drafting也三轮赢|同工作记录控制支持调度收益；lookahead+MTP早期出现一轮回退。保留opt-in|
|`--lookahead-swaps 1`（原2）|无drafting原prompt配对改善中位数0.837，第二prompt0.437；CPU-first关闭仍改善0.981。两个prompt MTP2也三轮赢|改变驻留、数值路径与工作量；MTP收益混有接受率变化。保留候选，不将总收益视为搬运提速|
|`NAIWA_SINGLE_GPU_SAMPLE=1`|单序列argmax通过现有资格检查后，使用已有GPU batch top1，而不是退回CPU logits。两个prompt无drafting改善中位数1.743/1.887；两个no-lookahead控制文本一致|诊断post-head间隙约1.89→0.55ms，head约1.11ms不变。MTP下无稳定增益。默认关闭，仍需边界/完整fallback验收|
|physical5 CPU pool|5worker+参与caller，物理核布局，20ms spin。首轮无drafting改善中位数3.661，MTP2改善2.704ms/token|dispatcher、线程数和affinity联合变化，非单纯spin收益；后续对照赢两轮输一轮。需第二prompt/chat及接受数诊断|

### 当前已测组合（本机专用实验参数）

```bash
NAIWA_CPU_BEFORE_DMA=1 \
NAIWA_SINGLE_GPU_SAMPLE=1 \
NAIWA_CPU_SPIN_US=20000 \
NAIWA_CPU_MOE_THREADS=5 \
NAIWA_CPU_PHYSICAL_LAYOUT=1 \
target/release/naiwa-rs serve \
  ...原有模型、缓存及服务参数... \
  --pcie-frac 0.4 --lookahead-swaps 1 \
  --mtp --mtp-n-predict 2
```

physical5绑定本机CPU1–5的worker及CPU0的caller，依赖本机6C/12T拓扑，不能直接推广到其他机器。无drafting省略 `--mtp` 与 `--mtp-n-predict`；不要为了显式深度误开启MTP。干净测试需清除ambient诊断变量；上方是参数示意，不是完整可运行命令。

## 4. 五轮同session对照（最终性能锚点）

|轮次|naiwa 当前组合，MTP2|Strata suffix0|Strata suffix3|
|---|---:|---:|---:|
|1|36.805|37.849|37.740|
|2|37.784|37.797|37.791|
|3|36.702|37.841|37.796|
|4|36.318|37.827|37.853|
|5|38.220|37.913|37.794|
|**进程中位数的中位数**|**36.805**|**37.841**|**37.794**|

单位ms/token。共15次进程启动、45个请求，无失败/重试；顺序ABC/BCA/CAB/ABC/BCA，5轮不是完全均衡顺序（6轮才完整）。

- naiwa与suffix3的中位数差 **−0.990ms/token（−2.6%）**；配对改善中位数 **0.935ms/token**，两个估计量不同。
- 赢4/5轮，第二轮近乎持平，第五轮回退0.426ms；naiwa范围36.318–38.220，Strata suffix3范围37.740–37.853。
- Strata `--spec 4 --suffix-draft 3 --lookup-chain 0`：每次重启三请求分别接受0/0、1/2、1/1，五轮共接受10/提议15，实际draft覆盖很少。suffix3是最短匹配长度，不是神经draft深度3。
- 当前prompt偏低重复度；不能外推到代码续写、长文本编辑或完整chat语料。
- MTP clean未采集逐请求接受数，不能把轮间波动直接归因于接受率或kernel成本。

**证据：** [五轮报告](../../harness/attribution/results/strata-suffix5.md)、对应manifest/report/draft日志；配置 `harness/attribution/strata-suffix-comparison-5round.json`；raw `/tmp/attr-strata-suffix5-01`。

## 5. 结构归因取得的进展

### 缓存策略：从“字节多”收敛到低利用填充

- budget2时cache-fill约169MB/48层等价窗口，约36.5%–38.0%的填充字节在下一次slot发布前没有形成resident hit；尾部仍驻留未使用另作右删失，不算浪费。
- budget2→1：cache-fill约169→91MB，demand staging约190→207MB，净减少约60MB。未使用即被替换的填充payload约减半。
- budget0字节最少但比1慢，说明目标是有效驻留与关键路径，不是最少字节。

### 采样路径：已找到可干预的控制流损失

- pipeline在通过计划检查后拒绝single-sequence/noncategorical；随后CPU logits路径占主要采样调用。
- opt-in仅放行现有argmax计划且不需要sampling logprob，保留recognizer/tool、penalty/DRY/bias/processors等现有检查。
- 独立诊断基线每进程384次single拒绝，候选不再命中；post-head间隙缩短约1.3ms。不能把CPUcopy时间、host sampler时间与GPU span相加。

### GPU/host账本：定位而非精确分摊

- Strata逐step GPU stream分段已闭合；naiwa补上专家区间及模型边界事件。
- 曾发现naiwa非MoE/推进、路由后提交/依赖和专家执行区间更长；但两边端点、外层开销不同，诊断差13.3ms不等于clean差9.73ms。
- mixer约naiwa7.18ms vs独立Strata6.39ms；HC mix细化到down/activation/up/reduce后只是局部逐层差异，暂不支持作为整个gap主因。
- LM head约1.11ms，Strata约1.19ms；输出投影不像主要差距。post-head fallback已产生实测改善。
- CUDA event密集探针曾增加约5.9ms，复用/后台输出后仍有扰动；18event/layer再次扰动1.55–3.03ms，稀疏采样才降至约0.24ms中位数。不能把探针成本逐桶扣除。

## 6. 负结果与未通过项

|实验/验收|结果与处置|
|---|---|
|8worker、不绑核、20ms自旋|当前组合下退化约10–11ms/token；不要单独打开spin却保留过量worker|
|shared expert简单提前到routed前|原prompt改善中位数0.181、第二0.484且一轮回退；实现撤回，补丁留存，不否定其他重叠方式|
|关闭promotion / lookahead|近期clean分别回退约1.246/0.550ms配对中位数；不作为默认调优|
|GPU fetch替代DMA、resident-first、更多waves|对应实现负结果/无稳定收益；详见M41/M42，不重复推广|
|GPU sampling三轮chat|TPOT与SSEp99改善，但TTFT三轮增加约23/183/346ms；各臂历史不同，需固定transcript控制|
|logprobs fallback|9对文本及非空logprobs完全一致，但性能回退配对中位数0.447ms；非回退门未过，原因未闭环|
|异步激活通道|mandatory read/skip/ack生命周期与正确性门未完成，不叠加推广|

## 7. 验证状态与后续门

**已完成：** 多组clean重启对照；两个no-lookahead prompt的GPU sampling文本一致；logprobs采集与数值对照；Strata实际draft日志检查；MTP诊断逐请求count/cost分解；独立探针扰动控制。近期引擎release build/check、Rust58通过/5忽略；harness Python15通过（各次执行详见原记录，非本次文档更新重跑）。

**尚未完成：**

- GPU sampling ties/非有限值、oracle、tool/penalty等完整fallback测试和默认晋升。
- physical5第二prompt、chat及接受率/固定工作量控制；CPU占用/拓扑适配不以TPOT单独替代验证。
- 固定transcript chat TTFT与fallback性能回退定位。
- 两引擎同request/step身份、GPU/host与client边界关联，完成剩余差距因果分摊。
- Strata神经MTP draft pack及双方真MTP对照。

**决策：** 保留三个代码机制/策略与physical5为显式实验组合，默认不变；五轮结果是当前锚点，不宣称稳定胜出、跨场景普适或整项M42完成。

## 8. 证据索引

- [CPU-first](../../harness/attribution/results/cpu-first.md)、[默认配置与MTP](../../harness/attribution/results/cpu-first-default-and-timeline.md)
- [缓存预算与生命周期](../../harness/attribution/results/fill-budget.md)、[跨prompt验证](../../harness/attribution/results/fill-budget-validation.md)
- [GPU sampling机制](../../harness/attribution/results/single-gpu-sampling.md)、[验证限制](../../harness/attribution/results/validation-followup.md)
- [CPU pool](../../harness/attribution/results/pool-tuning.md)、[早期同轮复验](../../harness/attribution/results/strata-best-pool.md)
- [结构对比](../../harness/attribution/results/structural-attribution.md)、[HC细化](../../harness/attribution/results/hc-projection-narrowing.md)、[尾部拆解](../../harness/attribution/results/head-narrowing.md)
- [Harness协议与限制](../../harness/attribution/README.md)；[逐项待办](../../harness/attribution/TODO.md)
