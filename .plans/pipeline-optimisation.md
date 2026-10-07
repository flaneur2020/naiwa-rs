# GPU expert cache — pipeline optimisation plan

## 背景和现状

当前实现（commit `26d4388`）已完成的内容：

- `--cpu-moe --expert-cache-gb N`：每层分配等份 slot，admission 在 prefill 时按命中计数填入 slot
- Decode 时按 hit/miss 分流，hit 走 `QMatMul` 在 slot device，miss 走 `MoEExperts::forward` 在 CPU
- 权重从不反量化（quantized bytes 原样拷贝）
- fork 新增 `expert_qtensors()` 暴露 QTensor，`forward_gather` top-k 改为从 `topk_ids` 实际维度读取

**当前未实现的关键流水线行为：**

1. Prefill 时，每层计算 attention 的同时，下一层所有未驻留 expert 就应该在后台被填入 slot
2. Decode 时，根据预测的 expert ids 提前异步填充 slot；如果预测时间内完成则 hit，否则 miss 走 CPU
3. GPU hit 和 CPU miss 真正并发执行（当前是顺序的）

---

## 不做的事

- PCIe streaming（miss expert 传到 GPU 算）：miss 永远在 CPU 算，不做 H2D + GPU GEMV for misses
- 大页内存 / `cudaHostRegister`：pinned 内存基础设施
- 以上对应 Strata 的 `pcie_num`、`PinnedArena` 路径

---

## 三个核心优化，按优先级排序

---

### P1 — Prefill 时后台并发填充下一层 slot

**Strata 对应**：`expert_source.cpp:2542` 的 `P.fetch(...)` — 在 CPU 开始算本层 expert 之前就发出 DMA，copy engine 和 CPU pool 并发跑。我们没有 PCIe streaming，但对应的行为是：在本层 attention（GPU）执行期间，后台线程把下一层所有未驻留 expert 的量化字节拷贝进 slot。

**机制**

`CachedLayer::forward` 的 prefill 路径（`num_tokens > 1`）目前是：

1. 收集本层路由的所有 expert ids，更新 hit_counts
2. 依次尝试填充空 slot（串行，占用 prefill 的时间）
3. CPU `MoEExperts::forward` 算完本层
4. 移到下一层

改为：

1. 收集 ids，更新 hit_counts
2. CPU `MoEExperts::forward` 算本层（不改变）
3. **同时**，向一个 per-layer 的后台 worker 提交"填充任务"：把所有未在任何 slot 里的 expert 按 hit_count 排序，依次填入空 slot。worker 用 `std::thread::scope` 或 `rayon` 跑，填充和下一层的 attention（GPU）重叠。
4. 在进入下一层 `forward` 之前，join worker（如果还没完成）

**两层之间的时序**：`forward` 在 mistral.rs 里是逐层串行调用的（decoder layer loop），每层的 `forward` 返回后才进下一层。这意味着"本层 forward 返回时，下一层 attention 还没开始"——重叠窗口是本层 forward 的 CPU expert 时间，不是 GPU attention 时间。要真正重叠进 GPU attention，需要 mistral.rs 的 decoder loop 在调用本层 MoE forward 时同时发起下一层 attention，这超出了 `RoutedExperts::forward` 接口的范围。

**实际可做的近似**：在本层 `forward` 内部，用 `rayon::spawn` 把"填充下一层 slot"的任务扔给线程池，`forward` 返回之后、下一层的 attention 开始之前，填充任务可能已经部分或全部完成。这不是完美重叠，但比现在的"prefill 结束后什么都没填"要好。

**需要的信息**：填充"下一层"需要知道下一层的 expert QTensors。`GpuCacheMoeProvider::build` 是在 load 时按层顺序调用的，每层的 `Arc<CachedLayer>` 可以在 provider 里按 layer index 存起来，让某一层在 forward 时能拿到 `layer+1` 的 cache 句柄来提交任务。

**footprint**：约 80 行，改 `expert_cache.rs`；需要 `GpuCacheMoeProvider` 保存 per-layer `Arc<CachedLayer>` 的列表，`CachedLayer` 加一个 `fill_all_empty_slots` 公开方法供跨层调用。

**风险**：`CachedLayer` 的 `load_slot` 用 unsafe 裸指针写，两个线程同时写不同 slot 是安全的（slot 独立，Vec 不 resize），但需要在注释里说清楚不变量，或改为 `Mutex<Option<Arc<QTensor>>>` per slot 彻底消除 unsafe。

---

### P2 — Decode 时 RouterLookahead 预测 + 异步预填充

**Strata 对应**：`RouterLookahead::submit` + `run()` 后台线程，用当前层激活值跑下一层 router（AVX2 点积），得到预测的 top-k ids，然后 `prefetch(layer+1, predicted_ids)` 把那些 expert 的页面暖进 L3。

**机制**

Decode 路径（1 token）：

1. 在 `CachedLayer::forward` 里，获取 `xs` 的激活值（当前层的输入，`[1, 1, hidden]`）
2. 提交到一个后台 lookahead 线程：拿下一层的 router 权重，跑点积，得到预测的 top-k expert ids（一次 BF16 matmul，约 2560×128=320KB，耗时 < 0.1 ms）
3. 对预测到的 expert ids，如果它们不在 slot 里，立即异步发起 `load_slot`（填充 `slot device` 上的 QTensor）
4. 本层 decode 正常执行（hit/miss 分流）
5. 下一层进来时，预测填充的 expert 大概率已经在 slot 里，变成 hit

**关键约束**：预测的 expert ids 是近似的（用当前层激活值近似下一层），不保证正确。错误预测：slot 里多了一个用不到的 expert，但不影响正确性（只是 slot 占用）。正确预测：本来是 miss 的 expert 提前进了 slot，变成 hit。

**需要的额外 API**：需要每层的 router 权重（`SparseMoeBlock::gate`，一个 `Linear` 层，`[num_experts, hidden_size]` BF16）。目前这是 private 的，需要在 fork 里加一个 accessor 或者在 `ExpertLoadContext` 里传入（load 时已经有 `vb`，可以读 `gate.weight`）。

**实现选择**：在 `GpuCacheMoeProvider::build` 里，从 `ctx.vb.pp("gate").get((num_experts, hidden_size), "weight")?` 读出 router 权重，存在 `CachedLayer` 里。这完全在现有 API 内，不需要 fork 改动。

**footprint**：约 150 行，改 `expert_cache.rs`；加一个 `struct RouterLookahead`（单个后台线程 + channel），`CachedLayer` 持有它，`forward` 里 decode 路径调用 `submit`。

---

### P3 — Decode hit/miss 真正并发

**Strata 对应**：`expert_source.cpp` 里 `P.fetch` 在 CPU 工作开始前发出，copy engine 和 CPU pool 并发跑。qflash-rs 里等价的是：GPU hit（`QMatMul::forward`）和 CPU miss（`MoEExperts::forward`）同时跑。

**当前问题**：`decode_forward` 是顺序的，先跑所有 hit，再跑所有 miss，再求和。

**解决方案**：`std::thread::scope` 把两路并发。阻碍是 slot 存储的 unsafe 裸指针——decode 路径只读（写只在 `load_slot`，发生在 admission），实际安全，但编译器不知道。

**两种修法**：

方案 A（最小改动）：给 `load_slot` 加一个 doc 注释说明"写只发生在锁内，decode 路径只读，两个 decode 线程不并发写"，在 `thread::scope` 里用 `unsafe` 显式 assert 这个不变量，加 `// SAFETY:` 注释。约 20 行。

方案 B（干净彻底）：把 `slot_gate/up/down` 从 `Vec<Option<Arc<QTensor>>>` 改为 `Vec<parking_lot::RwLock<Option<Arc<QTensor>>>>` 或 `Vec<arc_swap::ArcSwapOption<QTensor>>`（`arc-swap` crate，已是 mistral.rs 的间接依赖）。写用 `store`，读用 `load`，借用关系清晰，`thread::scope` 直接可用。约 40 行。

推荐方案 B：`arc-swap` crate footprint 小，读性能优于 `RwLock`，且已经在依赖树里。

**求和时的正确性**：权重只在最后 sum 时乘一次。当前代码里 hit 路径已经在 `run_slot` 里做了 `y * w`，miss 路径通过 `MoEExperts::forward` 内部应用了权重。两路的结果直接相加即可，不能再乘一次权重。

**footprint**：约 60 行（含 P3 的方案 B + `thread::scope`）。

---

## 执行顺序

P3 → P1 → P2，原因：

- P3 最小，无外部依赖，今天可做，是 P1 的前置（P1 里后台线程和 forward 主路径并发读 slot）
- P1 依赖 P3 的线程安全基础，加后台填充任务后需要重新过一遍借用
- P2 需要额外读 router 权重，改动最多，但不依赖 P1/P3

---

## 每项的验收标准

**P3**（在 M1 上就能验证）：
- `cargo check --all-targets` 干净
- `cargo test` 9 个测试全过
- decode 路径用 `std::thread::scope`，GPU 和 CPU 侧各自跑，结果求和正确

**P1**（需要真实模型 load，M1 上可用 `--cpu-moe` 无 GPU 测试）：
- Prefill 结束后，`slot_gate[i]` 非空的比例等于 `min(num_slots / num_experts, 1.0)`（每层）
- 两次 prefill 产生相同的 slot 分配（admission 是确定性的）
- 后台线程 panic 不会导致 forward 路径崩溃（join 时要捕获 panic）

**P2**（需要 CUDA 机器才能测 latency 收益，M1 上可测逻辑正确性）：
- 预测到的 expert ids 与真实路由的交集 ≥ random baseline（即 `RouteHitRate > 1/num_experts * top_k`）
- Lookahead 线程 panic 不传播（submit 是 fire-and-forget）
- Router 权重从 `ctx.vb` 读取成功，权重形状 `[num_experts, hidden_size]` 匹配 config

---

## 实现中的关键不变量（每项都必须在代码注释里说明）

1. **Slot 写入只在锁内发生**：`load_slot` 只在持有 `state` mutex 时（或通过 `arc-swap::store`）被调用；decode 路径只读 slot。
2. **权重只乘一次**：hit 路径的 `y * w` 和 miss 路径 `MoEExperts::forward` 内部的权重加权不能重复；两路结果直接相加。
3. **预测填充不影响正确性**：lookahead 填充的 expert 如果下一层没用到，slot 只是白占；如果用到了，hit 率提升，结果与不填充完全相同。
4. **后台任务 join 在 forward 返回前**：P1 里的 fill worker 必须在本层 `forward` 返回前 join，否则下一层的 `load_slot` 可能与 fill worker 的 `load_slot` 并发写同一个 slot。
