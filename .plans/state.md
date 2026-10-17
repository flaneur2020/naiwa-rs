# naiwa-rs v2 — state (2026-10-16)

Resume-here summary. Detail: `architecture-v2.md` (design), `roadmap.md` (plan),
`worklogs/` (evidence per milestone), `fork-inventory.md` (fork delta).

## 目标

换掉控制模型：**host 只发布数据、不发射**；一层 MoE 段捕获成一张 CUDA graph；GPU 在图内自旋等
CPU。依据：W105e 测得 decode 全程 **host-bound**（每层 GPU 已排空，dry 0.9–1.3 ms/layer）。

## 同步协议（定稿，`architecture-v2` §3.2）

```
mapped:  epoch u32（device 每 pass 写一次）  ready[K] u32（每点一个字，值=就绪到的 pass）
         payload[48][max_n] f32（host 写 CPU 结果；单块，无环无双缓冲）

节点:    e = *epoch ; spin while (ready[id] < e) ; __threadfence_system()
         CPU → acc += payload[layer] ; PRELOAD/STAGING → 直接继续
发布:    ready[id] = pass ；一律走 worker 的单条 FIFO 队列（含"无 miss 发零"）
常量:    id = layer*3 + kind , kind ∈ {PRELOAD=0, STAGING=1, CPU=2} , K = 144
```

三条必须记住的：

1. **`epoch` 只由设备写**（layer 0 处、图外发射一次 `bump_epoch`）→ host 只能用**自己的** pass
   计数，因此它读不到未来的 pass。host 若去读 epoch 会在 pass 边界阻塞甚至挂死。
2. **发布必须严格按点 FIFO**。payload 单块 + `ready` 单字：反序会让节点读到下一 pass 的数据，
   更糟的是让 `ready` **倒退**、留下永久等待。所以零值也必须走 worker 队列。
3. **单块 payload 的前提**：host 的 pass 推进被一个 GPU 产出的信号门控（今天 = 每层读 router 的
   doorbell，它在 pass k 全部层之后）。若哪天 token 线程不经 GPU 产出就能进下一 pass，就必须
   恢复双缓冲或加 `consumed` 握手。

## 里程碑

| | | 状态 |
|---|---|---|
| M0 | 零实参门铃原语（可见性 + 图内 replay） | ✅ 1000/1000、replay 200/200 |
| M1 | 静态布局 + 描述符驱动 kernel | ✅ 内核层逐位一致；engine `frac 0` 等于 oracle |
| M2 | per-layer MoE graph | ✅ 48 张捕获、1 launch/层；150 张捕获 11.6 ms |
| M3 | CPU 结果经图内节点回 GPU | ✅ 门禁 5/5 |
| M3.1 | doorbell 从 fork 移到引擎 | ✅ fork −1660 行 |
| M4 | `PRELOAD`/`STAGING` 两个门铃点 | 🔄 原语已落（`wait_ready` + `publish_point_on_stream`），接线未做 |
| M5 | lookahead 进专用 spin-then-park 线程 | ⬜ |
| M6 | PLE（等 Q2）+ 整步图 | ⬜ |

**M3 门禁**：`--pcie-frac 0` + `NAIWA_NO_LOOKAHEAD=1` → 5/5 逐字节等于冻结 oracle。
（开着 lookahead 时**分路本身**每步变化，那是既有的浮点顺序敏感性，需比变体集合而非单个参考。）

## 布局

```
~/code/naiwa-rs            引擎（package 名 naiwa-rs）+ .plans/ + harness/
  src/doorbell/{mod.rs,doorbell.cu}   Doorbell/FlagBuf/ScheduleBuf + 内核（自有 build.rs, cudaforge）
  src/expert_cache.rs       Bank/BankCapture、run_hits、decode_forward
  src/deferred.rs           CPU 专家 worker；start_schedule 发布到 schedule
  harness/m0/               M0 harness；harness/m1/capture_oracle.sh + reference.oracle.txt
~/code/mistral.rs-iq4xs    fork（IQ 格式内核 + 模型 plumbing；见 fork-inventory.md）
~/code/qflash-rs           旧的 eager oracle 二进制（只读，勿覆盖）
~/naiwa-rs/target          本 crate 的 target（.cargo/config.toml 固定）
~/bench-rel/naiwa-v2/      二进制、日志、harness 产物
```

## 已定的决策

- **不保留 A/B 环境开关**（只保留只读探针）。配置用 commit 记录。
- **专家批走 rayon，lookahead 走专用 spin-then-park 线程**（W94：池的专家半边输、lookahead 半边赢）。
- **混合 quant** → per-layer graph（不是一张图 replay 48 次）。
- **PCIe 不确定性是浮点顺序**，不是竞态（`NO_LOOKAHEAD` 下 10/10）。
- `MoeDecodeCapture` 留在 quant（要调 5–6 个私有 launcher，不为此开公共 API）。

## 踩过的坑（别再踩）

| 坑 | 结论 |
|---|---|
| host 写 mapped 的 per-step 值 | 跨步竞态；改成 device epoch + 静态期望值 |
| 捕获期禁用 host/device 拷贝、且 event 记录非法 | 把所有 guarded 指针解析挪到 `begin_capture` **之前**（fork 那套 toggle event tracking 的做法已弃用） |
| 捕获需要流空闲 | 捕获测试要 `--test-threads=1`/独占；引擎在 load 期捕获是安全的 |
| `bump_epoch` 传 host 指针 | 必须传 mapped 的 **device view**（UVA 下同址，单测看不出来） |
| 同步/异步发布混用 | 必须全部走 worker FIFO |

## 下一步（M4）

1. `stream_pcie` 的 DMA 入队后，在 **copy stream** 上 `publish_point_on_stream(id(layer,STAGING), pass)`；
2. lookahead 预载 DMA 同理（要把 `pass` 传给 worker）；
3. 每层图里加两个 `wait_ready` 节点（PRELOAD 在 compute cached 前，STAGING 在 compute DMA copied 前）；
4. 删掉 host 侧 `done.wait_on`（`cudaStreamWaitEvent`）；
5. 门禁：`--pcie-frac 0.55` + `NO_LOOKAHEAD` 与当前实现逐字节等价。

## 开着的

- **Q2（PLE 策略）**：M6 唯一前置；约半天可判（`ple.rs` 的 hash/prefetch 能否在 host 复现）。
- fork 的 W 时代探针（`PLE_DBG`/`MOE_DBG`/`MTP_DBG`，~150 行）可删——hygiene，不急。
