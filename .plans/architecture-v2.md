# naiwa-rs v2 — 架构对齐（读图 + 从零设计的机制）

记录于 2026-10-16。这是新仓库 `~/code/naiwa-rs` 的第一份文档：**只描述预期设计，不引用
`qflash-rs` 的任何代码结构**。`qflash-rs/.plans/{design,worklog,strata-gap-architecture,
spin-pool-doorbell}.md` 作为测量证据和历史结论保留引用（标为 `W###`），但实现从头写。

---

## 0. 为什么重写

W105e 的 `STEP_DBG` 探针给了决定性测量：稳态 decode **busy 0.000 ms/layer，dry 0.9–1.3 ms/layer**
——每个层边界 GPU 都已经排空流在等 host，48 层 × ~1 ms ≈ TPOT。即 decode 全程 **host-bound**。

qflash-rs 的问题不是某个 kernel 慢，而是**控制模型**：host 驱动 candle 的 eager stream，每层
经手 host 若干次（两次同步 D2H、一次挡住 stream 的 `cudaLaunchHostFunc`），并发射约 61 个 kernel、
做约 123 次分配（W66）。在这条链上做的任何局部优化（spin pool、host_works、segment、
round-robin）都被 W93–W98 证明为中性或负收益——因为它们要的前提是「host 很闲」，而我们的
host 恰恰最忙（池化的例外与拆分见 §6.3：lookahead 半边是赢的，专家半边是输的）。

v2 的目标是换掉控制模型本身：

> **GPU 侧自旋等 mapped 标志位；host 只做「数据」不再做「发射」；整个 MoE 段是一张 CUDA graph。**

---

## 1. 架构图的读解（逐节点）

图：`architecture.png`（作者手绘，2026-10-16，已随本目录提交）。

### 1.1 主链（虚线框内 = 一张 CUDA graph）

| # | 节点 | 我的读解 |
|---|---|---|
| 1 | Compute Layer 1 Attention（框外，绿） | 层内 attention/norm/router，**仍在 graph 之外**（见 §1.3） |
| 2 | Dispatch Experts（蓝） | MoE 分发：读 router 结果，按驻留状态切成「命中 / 未命中」，发布描述符 |
| 3 | ← "GPU check the CPU mapped seq of Preload Expert Cache" | dispatch 的第一步是**检查 mapped seq**：下一层（或本层）的 expert 预载是否就绪 |
| 4 | Compute cached GPU Experts（绿） | 驻留槽位在 VRAM，直接算 |
| 5 | "GPU Spins over the completion of DMA transfer result" | 自旋等 PCIe 那批专家 DMA 完成 |
| 6 | Compute DMA copied GPU Experts（绿） | 55% 的未命中经 PCIe 拷进 staging 后在此计算 |
| 7 | ← 框外 "DMA 55% Uncached Experts from host pinned memory" | copy stream 上的 DMA，来自 host pinned 的 expert 栈 |
| 8 | "a shared staging buffer for all layers for these DMAed experts" | **全局共享** staging（不是 per-layer）——这正是 W105 的 scratch ring 思路 |
| 9 | "GPU spins over the mapped seq for the completion of the CPU expert computation" | 自旋等 CPU 那 45% 的结果 |
| 10 | Gather MoE Result（蓝） | 命中 + PCIe + CPU 三路加权求和，出 MoE 输出 |
| 11 | "GPU copy from mapped result" | CPU 结果**从 mapped memory 拷回**，不是 host function 挡流 |

### 1.2 并行支路（虚线框下方 = CPU 侧）

| 节点 | 读解 |
|---|---|
| DMA 55% Uncached Experts from host pinned memory | 由 CPU 侧发起（"CPU spins doorbell get the list of uncached Experts to DMA to GPU"） |
| Compute 45% Uncached Experts in CPU | 同一批未命中的其余部分，**与 PCIe 那路并行** |
| "notify GPU about the completion of DMA" | GPU→host 方向的完成通知（用于 CPU 不重复处理同一 expert） |
| "notify GPU about the uncached result" | host→GPU：结果 + seq（第 11 步读取） |
| DMA preload the next layer's Expert Cache（黄） | lookahead 的槽位装填，**独立 copy stream** |
| "notify GPU about synced Expert Caches with a list" | host→GPU：新的驻留列表，供下一层 dispatch 分类（指向下一层的 Dispatch Experts） |
| 脚注："CPU spins doorbell get the list of uncached Experts to DMA to GPU; and compute in CPU in parallel; also, get the predicted experts of the next layer" | **不是一个池，而是按角色拆开**（§6.3）：lookahead/预测走专用 spin-then-park 线程，读 mapped 得到未命中列表；CPU 那部分未命中走 **rayon**；DMA 由 token 线程在 copy stream 上发起（worker 不发起 CUDA 调用，见 X5） |

### 1.3 三个关键读解（与 qflash-rs 现状的差异）

**(A) 虚线框只包一层，不是整个 48 层。** 框的左边界在 `Dispatch Experts`，右边界在
`Gather MoE Result`；`Compute Layer 2 Attention` 画在框**外**，然后才是下一轮的
`Dispatch Experts` 和 `......`。所以：

> **一张 graph = 一层的 MoE 段（dispatch → 驻留计算 → DMA 计算 → gather），每层的 attention
> 仍在 graph 外。每 step 有 48 次 graph launch。**

这与 `design.md` R5 的「整个 48 层窗口一张 graph、一次 launch」不同。差异来自 PLE：
`text.rs:985` 的 `ple.forward` 在**层与层之间**执行（不属于 MoE 段），而 PLE 表在 12 GB 卡上
不能常驻（`ple.rs:429 make_resident` 只在装得下时生效）→ host gather → `gathers_on_device()`
为假 → `text.rs:789 supports_decode_graphs()` 直接返回 false。**PLE 的 host gather 是整步 graph
的结构性阻碍，而图里没有 PLE 这个盒子。** 见 §3.3.4。

**但 PLE 比预期轻得多**（本轮核实）：`config.rs:299 ple()` 要求 `ple_layer_ids` **恰好一个**
元素，`layer_idx` 是它减一（测试断言 `ple.layer_idx == 1`）。即 **PLE 每 step 只执行一次**，
位置在第二个 decoder 层，一次算出所有层需要的 per-layer embedding（各层再用
`mapper.map(res, i)` 取自己的那份）。所以整步 graph 的阻碍不是「48 个 PLE 节点」，
而是**一个**节点——可解性的量级完全不同。

**(B) 数据通道是双向的，且 GPU 侧是自旋而不是阻塞。** 图上有三处 "GPU spins over the mapped seq"，
一处 "GPU check the CPU mapped seq"。当前的 fork 已经有 `doorbell_wait_seq_ge`（1 线程自旋 kernel）
和 `doorbell_wait_copy_f32`（等 seq 后从 mapped slot 拷进 device）——但**它们的 `target` 是 kernel
实参**，捕获进 graph 后会在 replay 时沿用捕获时刻的值。这是 §3.1 要解决的核心问题。

**(C) 三路分类的那一路（PCIe 55%）是 graph 内的 SM/DMA 拷贝，不是 host 决策。**
图里 "DMA 55% ... in parallel" 的发起者是 CPU 侧；完成通知走 mapped。所以 host 只发布**列表**，
拷贝本身在 stream 上。

---

## 2. 设计基线：per-layer 状态机

以一层为单位，三个角色：**token 线程（launch graph）**、**GPU（graph 内自旋）**、
**自旋池（N 个 worker + host 参与）**。

```
token 线程                      GPU（graph 内）                      自旋池
────────                        ───────────────                      ──────
queue 1 publish kernel ───────→ 写 x/ids/weights 进 mapped ring
(stream-ordered，层尾之后)       bump seq
queue graph launch  ─────────→
                                [node] wait seq(l)  ──→ host 侧 pool 被唤醒
                                                                     spin doorbell → 拿到
                                                                     未命中列表 + x
                                [node] compute resident slots        ├─ 发起 55% DMA
                                                                     └─ 算 45% (CPU)
                                [node] spin DMA done (seq)
                                [node] compute DMA'd experts
                                [node] spin CPU done (seq)  ←──────────────── publish 结果+seq
                                [node] gather
下一层 attention（eager）  ←────  graph 结束
```

关键不变量（沿用 W93–W106 已验证的部分）：
- **seq 单调递增，每层每方向一个**；payload 写在 seq bump 之前，`__threadfence_system()` 保证顺序。
- **ring depth ≥ 2**，防止慢读者与下一次写在同槽位相撞（`FlagBuf::RING_DEPTH` 现在是 8）。
- **一个 seq 只能被一个读者消费**；跨方向的等待必须同序（否则死锁）。
- **CPU 侧不做 CUDA 调用**（W99 的教训：worker 上的 CUDA 调用排到 stream 尾部，预测迟到，
  命中率 74% → 38%）。

---

## 3. 三项机制的设计

### 3.1 doorbell 同步原语：必须「零实参」

现有原语（fork `mistralrs-core/src/cuda/doorbell.cu`）：

| kernel | 方向 | 语义 |
|---|---|---|
| `doorbell_ring_kernel` | GPU→host | 写 ids/weights 进 mapped，fence，bump seq |
| `doorbell_copy_f32_kernel` | GPU→host | 写任意 f32 payload（= R2a 要的「发布 x」） |
| `doorbell_wait_seq_ge_kernel` | host→GPU | 1 线程自旋等 mapped seq ≥ target |
| `doorbell_wait_copy_f32_kernel` | host→GPU | 等 seq ≥ target，再从 mapped slot 拷进 device |

**问题：`target` 是 kernel 实参。** 捕获进 graph 时它被固化；replay 第 2 次起就错了。
cudarc 0.19.10 的安全 API 只有 `begin_capture` / `end_capture` / `launch` / `upload`
（`driver/safe/graph.rs:46,55,88,97`），**没有节点参数更新**（需要 `cuGraphExecKernelNodeSetParams`
的裸 sys 调用）。所以不改参数，改语义：

> **v2 原语约定：等值 kernel 只收指针，目标值放在 mapped memory 里，由 host 在 launch 前写。**

```cuda
// 实参只有指针 —— 同一张 graph 可以无限次 replay
__global__ void wait_ge_ptr(const volatile uint32_t* seq,   // mapped, device view
                           const volatile uint32_t* target) // mapped, host 写
{
    if (threadIdx.x == 0) {
        while (*seq < *target) __nanosleep(200);
        __threadfence_system();   // 保证 payload 可见
    }
}
```

`wait_copy_ptr` 同型：`(seq, target, payload, slot_off, dst, n)`，`target` 与 `slot_off` 都是
**指针/mapped 值**——`slot_off` 由 host 写进 mapped 的 per-layer 描述符，kernel 读它而不是收它。

**内存序（Q2 的答案，沿用 W93 harness 的 Ada 结论）**：
- GPU→host：payload 普通写 → `__threadfence_system()` → seq 用 **volatile atomic store**
  （host 用 `load(Acquire)` 读 seq，再用 `read_volatile` 读 payload）。
- host→GPU：payload 普通写 → `std::atomic::fence(Release)` → `write_volatile(seq)`；
  GPU 侧 `wait_ge_ptr` 自旋后 `__threadfence_system()` 再读 payload。
- RDNA 需要两次 fence；Ada 一次够（strata 注释）。**这是必须在目标卡上单独跑 harness 验证的
  第 1 件事**（10 行 × 200 次，100% 可见才算过）。

**SDMA 替代（可选，第二阶段）**：`cuStreamWaitValue32` 在 cudarc 的裸 sys 里可用
（`driver/sys/mod.rs:15992`）。它的 value 同样是 launch 期实参 → 捕获后固化，**不解决 graph 问题**，
但它几乎不占 SM。若 PLE/其它 host 等待点需要省 SM，可作为非 graph 路径的替代。

### 3.2 CPU → GPU 通知：设备 epoch 协议（取代动态 seq）

现状：`deferred` 已经不用 `cudaLaunchHostFunc`（W106 的 R2b 换成了 stream 上排队的 wait+copy
kernel），所以「token 线程不被阻塞」已经成立。剩下的问题是**这个等待进不了 graph**，而且
试过两种零实参写法都失败（详见 `worklogs/m3.md`）：

| 写法 | 失败方式 |
|---|---|
| `target`/`slot` 作 kernel 实参 | 捕获时被冻结 |
| host 写 mapped cell | **跨步竞态**：token 线程领先 GPU，节点读到下一步的 cell |
| 设备侧 `atomicAdd` 领号 | 原理成立，但要求「领到的号 = 发布者的 seq」严格 1:1；作业数浮动（只有有 miss 的层才发）就会漂移 |

**根因：`target` 是「每步变化的值」，而由 host 生产它必然与「host 领先 GPU」矛盾。**

#### 协议：把「第几步」的生产权交给设备

一块 mapped 分配（全局一份）：

```
epoch    : u32        ← 设备写、host 读
progress : u32        ← host 写、设备读
payload  : [R][max_n] f32   ← host 写、设备读（环）
```

- **设备生产 `epoch`**：`layer 0` 的图里有一个 prologue 节点，每 pass 执行一次
  `*epoch += 1; __threadfence_system();`。
- **host 生产 `progress`**：为同步点 `p` 写完 payload 后，普通写 → release fence →
  `*progress = p`（单调，只增）。
- **节点（图内）**：
  ```
  p = (*epoch) * N + L        // N = 每 pass 同步点数（常量），L = 层号（常量，烘在图里）
  while (*progress < p) __nanosleep(200);
  __threadfence_system();
  acc[i] += payload[(p % R) * max_n + i]
  ```
  节点实参只有**指针 + 常量** → 捕获期冻结毫无影响。

#### 为什么安全（唯一要记住的一条）

**`epoch` 只由设备写，host 只读** → host 永远读不到未来的 epoch，所以在编号上**只能落后或
持平，不可能领先**。落后是安全的：host 用旧 epoch 发布 `(k-1)*N+L` 时，设备的 target
`k*N+L` 更大，节点只是多等一会儿。**「读到下一步数据」这一整类竞态因此消失。**

层内重叠完全保留：host 在 pass k 开始时读到 `epoch=k`，就可以连续为 `L=1..48` 发布，比 GPU
在层上跑得更快；它只是不能比设备更早开始下一个 pass。

#### 一层的 CPU → GPU 通知点（命名与编号）

按 **GPU 命中它们的先后顺序**编号（顺序很重要，见不变量 7）：

| # | 常量 | 内容 | 谁生产 | 今天怎么实现 | 需要自旋？ |
|---|---|---|---|---|---|
| 0 | `SYNC_RESIDENCY` | 本层 dispatch 用的驻留列表 | host | stream 序 H2D 写描述符 | **否**（stream 已排序） |
| 1 | `SYNC_DMA_DONE` | 55% 那路 PCIe DMA 完成 | host 发起 / copy stream 执行 | copy-stream `CopyDone` + 主 stream 等待 | **否**（stream 序） |
| 2 | `SYNC_CPU_EXPERTS` | 45% 那路 CPU 专家的贡献 `[rows,hidden] f32` | CPU worker | `deferred` 的 mapped payload + seq | **是** |

**今天一层内真正需要 GPU 自旋的 CPU 通知只有一个：`SYNC_CPU_EXPERTS`。** 另两个是数据依赖，
靠 stream 顺序保证 —— 描述符是 `cudaMemcpyAsync` 排在 launch 之前，DMA 完成是 copy-stream
event；只有 host 直接写 mapped 才会出竞态（M1/M2 的描述符因此从未踩坑）。

编号方案（今天 `WAITS_PER_LAYER = 1`，其余两个只以注释/文档预留，不烧进 N/R）：

```rust
pub const SYNC_CPU_EXPERTS: u32 = 0;   // 今天唯一需要自旋的位置（层内第 0 位）
pub const WAITS_PER_LAYER:   u32 = 1;  // 将来若把 RESIDENCY/DMA_DONE 改成 mapped 通知再改大
pub const fn sync_id(layer: u32, k: u32) -> u32 { layer * WAITS_PER_LAYER + k }
pub const N: u32 = 48 * WAITS_PER_LAYER;   // 每 pass 的同步点总数
// R >= N + host 在 pass 内的领先量：N=48 取 R=128
```

节点烘入 `sync_id(layer, SYNC_CPU_EXPERTS)`，target = `epoch * N + sync_id(..)`。

#### 不变量

1. **同步点集合固定**：每层每 pass 都发布一次，即使 `n_cpu == 0` 也发布全零。这是静态排程的
   前提，也恰好消掉「同步点数浮动」这个耦合源。
2. **prologue 只在 layer 0 的图里**，每 pass 一次；stream 序保证它先于本 pass 其它层、后于
   上一 pass。
3. **`R ≥ N + 余量`**：`slot = p % R` 是按**点**算的，一个 pass 就有 N=48 个点。R=8 只够领先
   8 个点（8 层）。取 **R = 64**；更稳的做法是 host 读一个 mapped 的 `consumed` 做背压
   （发布 `p` 前要求 `consumed > p - R`），把「有界」从运气变成构造保证。
4. **失败也必须推进 `progress`**：worker 失败时发布零/空并把 `progress` 推到 `p`，否则 GPU
   永久自旋。
5. **u32 回绕**：`2^32 / N ≈ 89M` passes，够用。
6. **内存序**：设备写 epoch 用 `__threadfence_system()` 前后各一次；host 读 epoch 用 volatile
   + acquire；host 写 payload/progress 用 release fence；节点自旋后 `__threadfence_system()`。
7. **单标量 `progress` ⇒ 同步点全序。** 把 `progress` 推到 `epoch*N+id` 等价于宣告同一 pass
   内 id 更小的点全部就绪。所以 `sync_id` 的编号顺序必须同时等于 ① GPU 命中顺序 ② host 发布
   顺序。若将来某个等待点需要独立于其它点，必须把 `progress` 从 1 个字换成「每个等待点一个
   字」，否则会隐式跳过。

> 注意：这条协议**取代**了 §3.1 里「`target` 放 mapped」的写法。§3.1 的 `wait_ge_ptr` /
> `wait_copy_ptr` 仍然有用（它们只需要一个 host 写的目标值，适合 prefill 之类一次性路径），
> 但**每步重放的图**必须用本节的 epoch 协议。

### 3.3 CUDA graph 机制

#### 3.3.1 谁来 launch：**用 fork 已有的整套机制，不开新的**

fork `mistralrs-core/src/pipeline/cuda_graph.rs`（5058 行）已经实现了**整步 decode graph**：
`CudaDecodeGraphState` / `CudaDecodeGraphLaunch` / 按 batch bucket 的 key / 预捕获 /
`CudaGraphMemoryPoolGuard`（捕获期分配走内存池）/ `CudaGraphPinnedBuffer` +
`enable_cuda_graph_htod_cache`（捕获期小 H2D 缓存）。捕获入口在 `:2302`
（`stream.begin_capture(RELAXED)`），结束在 `:2336`（`end_capture` → `instantiate` → `upload`）。

> **结论：不要自己写 graph 捕获框架。** v2 的工作是**让我们的 MoE 段变成可捕获的区域**，
> 从而把 `supports_decode_graphs()` 的两道门打开。

#### 3.3.2 为什么现在捕获不了：两道门

```rust
// qwen4_exp/text.rs:789
pub fn supports_decode_graphs(&self) -> bool {
    self.layers.iter().filter_map(|l| l.ple.as_ref())
        .all(PleLayer::gathers_on_device)      // 门 1：PLE 必须在 device 上 gather
        && !self.has_external_experts()        // 门 2：外部 experts 一律否决
}
```

门 2 的注释就在 `moe_offload.rs:13`：「Models with externally owned experts disable CUDA decode
graph capture.」——因为当前的 provider 通过 `RoutedExperts::forward` 在 eager 路径上跑，
捕获不出正确的东西。

#### 3.3.3 捕获一个「变长命中集」

graph 的节点是固定的，但**每 step 的命中集合是变的**。三种可行解：

| 方案 | 做法 | 评价 |
|---|---|---|
| **A. 描述符驱动（推荐）** | kernel 的形状固定，**数量从 mapped/device 描述符里读**；host 在 launch 前把命中列表写进描述符 | 与 strata 的 "publish the plan (flag A)"（`expert_source.cpp:3235`）同构；一次捕获永久 replay |
| B. 捕获上界 + 掩码 | 按上界捕获 N 个节点，越界的 job 立即返回 | 节点数爆炸，graph 巨大 |
| C. 每次重捕获 | 命中集变化就重捕获 | 捕获本身比 eager 还贵，否决 |

方案 A 要求 `fused_moe_decode` 有一个**描述符驱动**的形态：现在是 host 算好 launch 几何后发射
（fork `install_moe_task_dispatcher` + `par_tasks`），要改成「读计数 → grid-stride 处理描述符数组」。
**这是最大的一块 kernel 侧工作，也是 R4 的自然归宿**（W101–W103 已经把 row kernel 做快了，
v2 要把它做成描述符驱动）。

**方案 A 的存在意义，就是让「每 step 变长的命中/未命中集」在同一个已捕获 graph 内被吃下——
不重捕获、更不回退 eager。** 这是 v2 相对 fork 现有整步 graph 的关键差别（也是一条不变量，
见 §6.2）。

#### 3.3.4 PLE：整步 graph 的唯一结构性阻碍

- PLE 表（Coder GGUF 里 IQ4_NL 26.8 GiB）在 12 GB 卡上**不可能常驻**（`ple.rs:429`）。
- PLE 在**每层 attention 之前**执行（`text.rs:985`），所以它不在 MoE 段里——不能被 §1.3(A) 的
  per-layer MoE graph 绕开，只能被从**整步** capture 里排除。
- 图里没有 PLE 盒子，等于承认了这一点：**门的形状决定了 graph 的形状。**

三条路线（需要在动手前定）：

1. **接受现状**：graph 只覆盖 MoE 段（图上画的）；attention + PLE 保持 eager。
   → 每 step 48 次 launch + 每层 attention 的 host 往返仍在。
   → 收益：MoE 段的 ~60 个 kernel 发射 + 2 次同步消失。**但 host 往返未必消失**（attention 还
     要 host 发射）。这是 W105e dry 桶的一部分，必须诚实标注为**部分收益**。
2. **PLE 预取到 pre-graph 阶段**：decode 的下一步 token id 在**本步开始前已知**，所以 host 可以在
   launch 前把该 token 需要的 16 行 gather 好、上传到固定的 device staging buffer（在捕获之外，
   普通 H2D），graph 内只做索引 + 加成。
   → 需要确认：n-gram hash 依赖 per-layer 的 conv history（device 上的循环状态）能否在 host 复现。
     若不能，这个方案要改 hash 的所有权（fork 改动）。
3. **缩小表编码 / 热点常驻**：把热点行做成 device 侧 hash 索引缓存，miss 走一个「上一步预取」的
   固定槽位；永远保留一步的余量。复杂度最高，但唯一能让 PLE 进 graph 又能容忍 miss 的形态。

#### 3.3.5 捕获期的硬约束（candle 侧）

- **host→device 拷贝在捕获期被禁止**（`check_capture_copy`），除非走
  `CUDA_GRAPH_HTOD_CACHE`（缓存首次上传、replay 复用同一 device 切片）。→ 一切输入必须
  **预先 stage 到固定 buffer**，或用指针参数从固定 buffer 读。
- **分配在捕获期**：candle 允许（走 `CudaGraphMemoryPoolGuard`），但要求每次 replay 形状一致。
  → 我们仍然选择**静态预分配**：expert cache 的 bank、staging ring、FlagBuf、描述符全部
  load 期一次性分配，地址在 graph 的整个生命周期内不变。
- **捕获期不能有 host 依赖的控制流**：`to_vec` / `to_device(Cpu)` / 任何 `dev_synchronize`
  都不允许出现在捕获路径上。→ 正好是 v2 要删掉的东西（W100 的 ② 和 ⑥）。

### 3.4 静态地址布局（graph 的前提）

一层的指针一旦固化，就永远不能变。加载顺序因此被固定：**load → 7-ring 预热（一次性建齐 7 个
shape 的 staging ring 与所有 per-layer bank）→ 按 `MoEGraphKey` 捕获 → 服务**。预热必须在捕获
之前：W105 bug (a) 证明 load 期连续分配会 OOM（11.6 GiB 空闲下 2/3 概率），ring 只能惰性创建，
而 graph 要求地址在捕获时已经有效。因此：

```
[VRAM]
  expert_banks      : per-layer bank（Q3=(b)：48+MTP 张 graph，逐层基址烘死在各自 graph 里）
  staging_ring      : 全局共享（图上明说），per-shape 分组（W105：7 个形状族，混合 quant）
  flag_gpu_to_host  : per-layer，ring depth 8  （ids, weights, x）
  flag_host_to_gpu  : per-layer，ring depth 8  （命中描述符, DMA 列表, CPU 结果, 驻留列表）
  desc_{in,out}     : per-layer，计数 + 索引数组
[host pinned]
  expert stacks     : 每层三投影，`cuMemHostRegister(PORTABLE)`
```

---

## 4. fork 的切入点

### 4.1 mistral.rs（`../mistral.rs-iq4xs`，保持最小）

| # | 切入点 | 改动 | 为什么必须 |
|---|---|---|---|
| F1 | `supports_decode_graphs()`（`qwen4_exp/text.rs:789`） | 门 2 改成「provider 声明自己可捕获」而不是无条件 `!has_external_experts()` | 否则 graph 永远不启动 |
| F2 | `RoutedExperts` / `RoutedExpertsProvider`（`mistralrs-core/src/moe_offload.rs`） | 从「提供 experts」扩展为「提供 **MoE 段的可捕获 emitter**」：emitter 发射 dispatch→wait→copy→decode→gather 的节点序列。**注意**：Q1=(c) 不做整步捕获，所以 emitter 是**每个 site 在自己的 per-layer 捕获会话里被调用一次**，而不是在一次整步捕获里被调用 48 次 | 现在 provider 只能在 eager 路径被调用 |
| F3 | `cuda/doorbell.cu` + `doorbell.rs` | 加 `wait_ge_ptr` / `wait_copy_ptr`（指针版）；`FlagBuf` 的 target 描述符化管理 | §3.1 的零实参要求 |
| F4 | **`mistralrs-quant` 的 `indexed_moe_fused_decode`**（GPU 路径，`gguf/cuda.rs:1586`）——**不是** CPU 的 `fused_moe_decode`（`fused_cpu.rs`，那是 `/par_tasks` 的 CPU 派发，跟捕获无关） | 描述符驱动 / 定长网格形态：`MoeDecodeCapture` 用自有的一次性缓冲 + 固定网格 `(⌈n/8⌉, max_topk, max_batch)`，计数由填好的 ids/weights 描述符表达 | §3.3.3 方案 A。已落地（M1.1，fork `729d2c64`） |
| F5 | PLE（若走 §3.3.4 路线 2/3） | hash 所有权 / 预取接口 | 唯一能打开整步 graph 的路 |

现有可复用的（不要重写）：`pipeline/cuda_graph.rs` 全套、`moa_router_topk`、`indexed_copy`、
`moe_gemv`/`moe_gemm`。

### 4.2 candle（`../candle`，尽量不动）

| # | 切入点 | 判断 |
|---|---|---|
| C1 | 捕获期 H2D 禁令 + `CUDA_GRAPH_HTOD_CACHE` | **已具备，够用**（`device.rs:92,159`）；只要我们把输入预 stage 到固定 buffer |
| C2 | 捕获期分配 | **已具备**（内存池 guard） |
| C3 | 需要一个「裸流」入口做 `begin_capture` | **已具备**（`cuda_stream()`） |
| C4 | `CudaStream::begin_capture` / `end_capture` / `CudaGraph::launch` | **已具备**（cudarc 0.19.10 `driver/safe/graph.rs`） |

> **结论：candle 侧预计零改动。** 前提是我们的设计遵守「捕获路径上无 host 依赖控制流、
> 无临时分配、无 H2D」。若发现需要改，先怀疑设计而不是先改 candle。

---

## 5. 与现有结论的冲突（必须一起解决的）

| # | 冲突 | 处理 |
|---|---|---|
| X1 | W105b：`pcie_frac` 0 vs 0.55 的 TPOT 是 48.8 vs 49.1 **平手**（CPU miss 已被 `deferred` 全藏住）。图上仍画 55/45 分路 | graph 世界里意义变了：PCIe 那路**在 graph 内**、零 host 参与；CPU 那路靠 seq 自旋。分路比例不再影响「谁来等」，只影响 DRAM/PCIe 带宽分配 → **保留这个旋钮，但不再当作 TPOT 杠杆** |
| X2 | W105f：np3/np4 无收益，np2 已饱和 | v2 的设计按 **np2（3 rows）** 定型；行数不影响节点形状（描述符驱动），但 G=2 分组流水（R6）留作后续 |
| X3 | W100：第一次 D2H 的 x 与第二次 `to_device(Cpu)` 重复 | 图上已经解决：x 只经 mapped 通道走一次（"GPU copy from mapped result" 是**回程**） |
| X4 | design.md R5 说整步 graph + 每层 kernel 降到 20 多个 | 与 §1.3(A) 的 per-layer MoE graph **不是同一件事**。图是可行的第一步；整步 graph 需要先解 PLE |
| X5 | W99：worker 上的 CUDA 调用会迟到 | 图上的 CPU 支路只做**列清单 + CPU 计算 + 预测**；自旋池 worker **不发起任何 CUDA 调用**。55% 那路是真 H2D（pinned→VRAM），捕获期禁 H2D，因此必须在 graph 外由 **token 线程（或一个专职 copy-issuer 线程）**在 copy stream 上 `cudaMemcpyAsync`；worker 只把「未命中列表」写进 mapped 让 token 线程看到。文档初稿「CPU 支路做 DMA 发起」的措辞在此修正 |
| X6 | W94：spin pool 「loses stably」，但同一条 finding #3 说 lookahead 半边是 **win** | 拆开处理：专家批保持 rayon、lookahead 走专用 spin-then-park 线程（§6.3）。W94 的失败机制（host 在 `drain_epoch` 上的关键路径 + per-call publish/fence）正是 M3 要删的东西，所以它既不能证成也不能证伪 v2 里的池；而 lookahead 的收益已实测 |

---

## 6. 决策记录

2026-10-16 对齐结论。Q1/Q3/Q4/Q5 已定；Q2 仍开放。

| # | 问题 | 结论 |
|---|---|---|
| Q1 | graph 的范围 | **(c)** 先做 per-layer MoE 段 graph（48 次 launch，图上画的）；PLE 单独立项，之后视 PLE 结果再升级到整步 (b) |
| Q2 | PLE 的策略 | **仍开放**。上一轮已把天平显著推向 (b)：PLE 每 step 只有一个节点（§1.3(A)），host 预取到 pre-graph staging 的性价比远高于文档初稿的估计；唯一 blocker 是 n-gram hash 依赖 device 上的 per-layer conv history 能否在 host 复现——应在 M0–M2 期间用半天验证 |
| Q3 | expert bank 布局 | **(b)** per-layer bank + 48 张 graph（+ MTP 头 1 张）。理由见 §6.1 |
| Q4 | 第一个里程碑 | **M0**（见 §7）：零实参原语的可见性 + replay 200 次，是硬阻塞项 |
| Q5 | CPU/PCIe 分路比例 | **动态**，按 miss 数决定；`frac` 只作为带宽旋钮，不再当 TPOT 杠杆（X1） |

### 6.1 Q3=(b) 的理由：混合 quant

本轮核实（W105 bug (c)）：该模型**确实是混合 quant**——各层 expert 的 per-projection 字节数与
dtype 不一致，实测 **7 种 expert shape**（IQ3_XXS ×2、IQ2_S ×2、IQ3_S ×2、IQ4XS；如 524800
与 627200、921600 与 460800 字节的 expert 并存），`expert_cache.rs:897` 的
`ScratchKey = ([usize; 3], u8)` 已经把它编码进去（甚至同一层 gate/up/down 三投影也可不同）。

- **Q3(a) 被它直接否决**：一张捕获 graph 的 kernel 节点只能固定一种 dtype/几何，覆盖不了 7 种。
- **Q3(b) 天然吃下**：每层的 graph 在**捕获期**钉死自己的 quant instantiation，replay 零运行时
  分支、零 warp divergence。
- 措辞修正：Q3(b) 原文「48 张**结构相同**」不准确，应为「48+MTP 张 graph，节点拓扑按 shape
  group 只有 ~7 种 variant，逐层的 bank 基址/描述符/staging 指针不同」。
- 共享结构（`FlagBuf`、描述符 schema、全局 staging ring）必须 **format-agnostic**：只放
  count/offset/pointer，格式几何全部烘进各层 graph。**这是 F4 的硬性规则。**
- staging ring 保持 shape-keyed（已有 7 个 ring）；slot offset 继续走描述符/device 计数，不能烘死
  （否则与 `ScratchRing.cursor` 的 round-robin 冲突）。
- 驻留路径与 DMA/staging 路径共用同一 quant 格式（cache 存 raw expert bytes），所以一张 graph 内
  「compute cached」与「compute DMA copied」两节点格式一致，无需格式转换。

### 6.2 捕获 key 与捕获时机

fork 现有的 `CudaDecodeGraphKey`（`cuda_graph.rs:924`）带 context-len / recurrent kind /
decode_rows / 全输入张量形状等维度，捕获单元是**整步 decode**。若把 `layer` 直接塞进去，
图数量 = (现有 key 数) × 48，会为 attention-only 维度付出 48 倍捕获成本。

**结论：per-layer MoE graph 用另起的窄 key，不复用整步 registry 的语义。**

```
MoEGraphKey = (device, site_index, rows_bucket)
  site_index  = 48 个 decoder 层 + 1 个 MTP 头（ExpertSite.is_mtp）
  rows_bucket = np0/np1/np2 → 3
```

数量 ≈ 49 × 3 ≈ **150 张**。context length / KV layout / recurrent kind 不影响 MoE 段形状，不进
key；命中的**数量**由描述符驱动，也不进 key。因为 §3.3.5 选了静态预分配，这些图在捕获期
**零分配、零 H2D**，不消耗 `CudaGraphMemoryPoolScope`（`:64`）也不碰 `CudaGraphHtodCache`。

实现选择：**另起一个小的 MoE registry**（`HashMap<MoEGraphKey, CudaGraphLaunch>`，只复用
`stream.begin_capture` / `end_capture` / `instantiate` / `upload` 原语，约 100 行），**不**把
`site_index` 塞进整步 `cuda_graph.rs`——整步 registry 的 pad-row KV/recurrent 写入、spec state
admission、flashinfer scratch 对 MoE 段全是噪音。因此 §3.3.1 的「不要自己写 graph 捕获框架」
限定为「不要重写整步框架」。

捕获时机（用户已定，与 §3.4 的加载顺序一致）：
1. load 完成后、任何捕获之前，先跑 **7-ring 预热**，把 7 个 shape 的 staging ring 与所有
   per-layer bank 一次性建好。
2. 再按 `MoEGraphKey` 捕获。M2 需实测 150 次 `begin_capture→instantiate→upload` 的总耗时；
   若偏大，退路是只预捕获稳态 bucket（np2），np0/np1 保持 eager。**注意这个退路是「行数
   bucket」回退，与 miss 无关**（attention 本来就在 graph 外，回退路径天然存在）。
3. **pad rows**：整步框架的 pad 行别名到 row 0；MoE 段里 pad 行会产生多余的专家计算与描述符项。
   描述符必须带 `real_rows`，DMA/命中列表在 host 侧按 `real_rows` 生成（pad 行不进列表）。

**不变量（本轮澄清）：miss 永不回退 eager。** 变长的命中集是常态（W105：hit 77–93%），必须由
§3.3.3 方案 A 的**描述符驱动 kernel** 在同一个已捕获 graph 内吃下：命中走驻留槽位，未命中走
PCIe（graph 内自旋等 DMA，§1.1 node 5–6）和 CPU（graph 内自旋等 mapped 结果再 D2D 拷回，
§3.2），三路在 gather 节点求和。**唯一允许的 eager 回退是「行数超出已捕获的 `rows_bucket`」**，
且惰性捕获该 bucket 即可消掉。上面第 2 点的 np0/np1 属于这一类，**不是** miss 回退。

### 6.3 专家批的派发：rayon vs spin pool

**结论：按角色拆开，不二选一。** 专家批保持 **rayon**（fork 的 `install_moe_task_dispatcher`
默认）；lookahead/预测走 **1–2 个专用 spin-then-park 线程**（照 strata `kSpinBeforeSleep`，自旋窗口
按层间隔 ~560 µs）。固定为这一种配置，**不保留 A/B 开关**（§6.4）。

依据（W94，18 runs；胜负在同一次实验的两半里就分好了）：

- **反对池的理由只落在专家批半边**：np2 下池 −19%~−25%，而 np0（小批、多次小调用）几乎打平。
  W94 finding #2：损失**随每次调用工作量增长**而非调用频率——是 host 在 `drain_epoch` 上的关键
  路径（发布线程自己 drain 完整个 segment 才 `wait_done`）加 per-call publish/fence，而 rayon 把
  这些摊薄了；**不是** claim 争用（W94 换成 segment dispatch 反而更差；W95 的 round-robin 只收回
  5.6 里的 0.6）。
- **支持池的理由只落在 lookahead 半边**：W94 finding #3「The lookahead half of the pool is a win,
  still」，命中率每个 cell 都更高（最好 75.8%）。
- 所以 §1.2 脚注的「一个自旋池，三件事」合并错了：两半的需求相反——专家要吞吐/负载均衡（rayon
  的强项，也天然吃下混合 quant 的 7 种 shape 成本差异），lookahead 要最低延迟且**不能排队在专家
  任务之后**（专用线程的强项）。

**v2 改变了什么**：M3 删掉了 `drain_epoch` / 发布线程 drain / per-call host fence，即 W94 的失败
机制本身。因此 **W94 不能作为 v2 里「池会输」的证据**；但它同样**不是「池会赢」的证据**——池唯一
已证的优势在 lookahead 半边。专家批没有理由改用池，而池的持续空转会直接消耗 v2 想腾出来的 host
资源。

**派发配置在 v2 里固定**为 rayon + 专用 lookahead 线程（§6.4，无 A/B 开关）。M5 仍要用
STEP_DBG 测 (a) 专家派发是否在关键路径上、(b) 命中率是否跌破 W105 的 ~78% 拐点；这是**验证**
既定配置，不是翻开关。若 (a) 反常成立，把它当作一个需要单独处理的测量问题（临时改代码验证），
而不是常备一条分叉。

### 6.4 不保留 A/B 环境开关

**结论：v2 不保留 A/B 环境开关。** 每个决策点只实现选定的一种配置（专家批 rayon、lookahead
专用 spin-then-park 线程、per-layer MoE graph 等），把分支和死路径从代码里去掉。

理由：qflash-rs 的开关（`NAIWA_SPIN_POOL` / `NAIWA_HOST_WORKS` / `NAIWA_LH_MODE` …）是为「在
同一个 binary 上反复证伪」服务的，代价是每个行为都挂一条永久分叉；到 W93–W98 时这些分叉本身
已经成了复杂度。而 v2 的两项前置结论（W94 分半、W105b CPU 路径全隐藏）已经把派发选择定住，
再为「万一要翻案」保留开关，收益不抵复杂度。

代价与接受方式：v2 无法在不改代码的情况下翻臂。若 M5 的 STEP_DBG 出现与预期相反的异常
（专家派发确实在关键路径上），处理方式是**临时改代码验证**，而不是常备一个开关；此时以 git
commit 为单位记录两侧配置（benchmark 记录 commit hash，而不是 env 组合）。

**仍然保留：只读探针。** `NAIWA_STEP_DBG` 这类 env 门控的日志/事件插桩**不是** A/B 开关——M5
的判据（W105e 的 dry bucket 测量）依赖它，且它不改行为、无分叉。

### 6.5 M1 落地修正（实现期，2026-10-16）

- **F4 的目标是 GPU 的 `indexed_moe_fused_decode`，不是 CPU 的 `fused_moe_decode`。** 原表写错
  了（§4.1 已改）。GPU 路径是 `expert_cache.rs:1145 fused_decode → indexed_moe_fused_decode`，
  四个 kernel，网格由 **value 实参**算出（`dim3 grid(⌈n/8⌉, topk, batch)`）——所以「传入
  `batch=max_batch, topk=max_topk` 就得到固定网格」，**无需改 kernel**。
- **V1–V3 是被「绕开」而不是「修好」的。** 捕获的三个阻碍（全局 workspace 会增长并 retire、
  `alloc_zeros` 每调用一次分配、gate+up workspace 随 `batch*topk` 变）都属旧路径；
  `MoeDecodeCapture` 用**自有一次性缓冲**，因此 v2 全程走它时这些路径不进 v2。旧
  `indexed_moe_fused_decode` 保持不变（additive），eager 路径零回归。
- **`MoeDecodeCapture::output()` 是指定缓冲的别名**，只在下次 `run` 前有效。这是 graph 的语义
  （图拥有该缓冲）；eager 调用方必须自己 copy。不是缺陷，是设计。
- **sentinel guard 推到 M2。** `moe_gemv_fused_gate_up_impl` 没有 `expert_id` 越界检查（forward
  变体有），所以 M1 用「valid slot + weight 0」填充（qflash 既有做法）。加 sentinel
  （`expert_id >= num_experts` 写零 + launcher/FFI 增加 `num_experts`）推迟到 M2 真正需要它时。

---

## 7. 实施顺序（每步可行、可独立验证）

**共同判据**：
- **数值 oracle**：冻结的 eager 参考——固定 prompt 集 + greedy，逐 token 比对。
- **结构性判据**：发射计数 / 捕获计数 / 分配计数 / dry bucket。它不依赖性能数字，因此在 8% 噪声里
  也能判定通过与否（性能另记，不作为门槛）。
- 每个里程碑只引入一类新机制，保证失败时归因单一。

**M0 — 同步原语 harness（~1 天，独立于引擎）— ✅ DONE 2026-10-16（见 `worklogs/m0.md`）**
- 内容：零实参 `wait_ge_ptr` / `wait_copy_ptr` 的 ① 跨 host/GPU 可见性（目标卡 fence 语义：Ada
  一次、RDNA 两次）② 捕获进 graph 后 replay 200 次不丢。
- 依赖：无（**硬阻塞项**）。
- 判据：① 1000 次循环 100% 可见；② 捕获后 replay 200 次 payload 全部正确；③ **内核确实自旋**
  （`waited == iters/replays`）。
- 产物：源码 `harness/m0/`（kernel + 四臂 fence 对照 + must-fail 的 baking 反证）；二进制与日志
  `~/bench-rel/naiwa-v2/m0/`。
- 可行性：纯 harness，不碰引擎。
- 结果：全过（RTX 4070 / Ada）。注意：Ada 上两个 fence 对此 pattern 均非必需（保留以备 RDNA）；
  且**数值正确 ≠ 未烘焙**——baking 对照里值仍全对，只有 `waited` 抓到它。

**M1 — 静态地址布局 + 描述符驱动 kernel（不做 graph）**
- 内容：① load → 7-ring 预热（7 个 shape 的 staging ring + 所有 per-layer bank 一次建齐）；
  ② `FlagBuf`/描述符全部 load 期分配，地址终身不变；③ `fused_moe_decode` 改为「读计数 →
  grid-stride 处理描述符数组」（F4）。
- 依赖：M0（原语）+ F4。
- 判据：① 预热后整个 decode 无新分配、地址不变（断言）；② **eager 下**老几何发射 vs 描述符驱动
  逐 token 数值一致，且命中集每步变化；③ 不引入 graph/PCIe 的任何东西。
- 为什么可独立验证：**完全不碰捕获**，用现成 eager 路径就能测——把「最大的一块 kernel 工作」
  （W101–W103 的延续）与 graph 机制解耦；失败就只是 kernel 回归，不牵连架构。

**M2 — per-layer MoE graph 骨架（只跑驻留路径）**
- 内容：MoE registry（`MoEGraphKey`，§6.2）+ F2 emitter + per-layer 捕获/launch。
- 依赖：M1。
- 判据：① 全驻留配置下 greedy 逐 token == eager；② 一层从 ~60 次发射 → **1 次 launch、0 次自旋**；
  ③ 150 次 `begin_capture→instantiate→upload` 一次性成本已测并记录；④ pad rows 被 `real_rows` 排除。
- 备注：这是**脚手架范围，不是目标行为**；miss 由 M3/M4 加进图，绝不回退 eager（§6.2）。

**M3 — CPU 结果回程进 graph（miss 进入 graph）**
- 内容：`deferred` 的 `cudaLaunchHostFunc` → `wait_copy_ptr`（§3.2）。
- 依赖：M2。
- 判据：① 有 miss 时 greedy == eager；② **miss 全程在图内**（断言/计数证明未走 eager 路径）；
  ③ token 线程零同步点（STEP_DBG 的 dry bucket 下降）；④ 失败注入：worker 出错仍 bump seq，GPU
  不永久自旋（`Drop` 补 publish）。

**M4 — PCIe 路进 graph（三路齐）**
- 内容：55% 那路（真 H2D，graph 外由 token 线程发起；graph 内自旋等 DMA 完成）入图。
- 依赖：M3。
- 判据：① 命中/PCIe/CPU 三路求和 == eager；② **任何 miss 组合都不重捕获**（捕获计数不随
  miss/hit 变化）；③ frac 只作带宽旋钮（X1/Q5）。

**M5 — lookahead 专用 spin-then-park 线程 + 验证派发假设**
- 内容：lookahead/预测进 1–2 个专用线程（§6.3）；专家批保持 rayon。
- 依赖：**实现**只依赖 M2；**测量**依赖 M4（要 host 已从关键路径移除才公平）。
- 判据：① 命中率不低于现状（不跌破 W105 ~78% 拐点）；② STEP_DBG 证明专家派发不在关键路径上；
  ③ 无 A/B 开关（§6.4），以 commit 记录配置。
- 备注：若 ② 反常成立，临时改代码单独评估，不加常备分叉。

**M6 — PLE（Q2 定）+ 整步 graph**
- 依赖：Q2 结论 + F1 + F5。
- 判据：整步 **1 次 launch**；greedy == eager；捕获路径无 host 依赖控制流、无临时分配、无 H2D。

**独立验证矩阵**（每步只新增一类机制）：

| 步 | 新增机制 | 独立判据（不靠性能） |
|---|---|---|
| M0 | mapped 自旋原语 | 可见性 1000/1000；replay 200 次不丢 |
| M1 | 静态布局 + 描述符 kernel | eager 数值一致；无新分配/地址不变 |
| M2 | per-layer 捕获 | 发射 60→1；逐 token == eager |
| M3 | CPU 回程进图 | miss 在图内（计数）；dry bucket 下降 |
| M4 | PCIe 进图 | 三路 == eager；捕获计数与 miss 无关 |
| M5 | lookahead 专用线程 | 命中率不破拐点；派发不在关键路径 |
| M6 | PLE + 整步图 | 1 次 launch；逐 token == eager |

---

## 附：证据索引（本轮核实，非转述）

| 事实 | 位置 |
|---|---|
| 外部 experts 否决 graph 捕获 | `mistralrs-core/src/vision_models/qwen4_exp/text.rs:789`；`mistralrs-core/src/moe_offload.rs:13`（不在 `qwen4_exp/` 下） |
| 混合 quant：7 种 expert shape | W105 bug (c)（`worklog.md` 4993 段附近）；`qflash-rs/src/expert_cache.rs:897`（`ScratchKey`） |
| PLE 在每层 attention 之前 | `qwen4_exp/text.rs:985–1010` |
| PLE 表只在装得下时常驻 | `qwen4_exp/ple.rs:429`（`make_resident`）、`:862`（`gathers_on_device`） |
| 已有 4 个 doorbell kernel，target 为实参 | `mistralrs-core/src/cuda/doorbell.cu` |
| `FlagBuf` ring depth 8 / 双向 API | `mistralrs-core/src/cuda/doorbell.rs:201,218,287,359,425,457` |
| 整步 decode graph 框架 | `mistralrs-core/src/pipeline/cuda_graph.rs:2302,2336,2420,3114` |
| 捕获期禁 H2D | `candle-core/src/cuda_backend/device.rs:159`；缓存路径 `:92,192` |
| cudarc graph 安全 API（无节点参数更新） | `cudarc-0.19.10/src/driver/safe/graph.rs:46,55,88,97,110,122` |
| `cuStreamWaitValue32` 可用（裸 sys） | `cudarc-0.19.10/src/driver/sys/mod.rs:15992` |
| 描述符驱动 / 发布执行计划的先例 | strata `src/core/expert_source.cpp:3235`（转引 W100/W105） |
