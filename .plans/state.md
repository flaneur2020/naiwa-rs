# naiwa-rs v2 — state (2026-10-17, M5 done)

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
发布:    ready[id] = pass。生产者有两种，都保证"数据先到":
         ① CPU 结果：worker 单条 FIFO 队列（含"无 miss 发零"），host 写 payload 后写 ready
         ② copy stream（M4）：cuStreamWriteValue32，排在 DMA 之后，host 不等待、不记事件
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
| M4 | PCIe 路进图（`STAGING`） | ✅ 5/5 + 2/2 逐字节；48 gate / 384 replay / 0 退化 |
| M5 | 专家回 rayon + 预测单线程事件驱动 + 异步预载 | ✅ 命中 81–84%、TPOT 78–87 ms、11/11 逐字节 |
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
| `copy_stream` 存在 ≠ staging 在服务 | `frac 0` 下 copy stream 仍存在（缓存槽靠它填），gate 若按 `copy_stream.is_some()` 判据就会等一个永不发布的点 → **挂死**。三处(捕获/发布/回放)必须同一个判据 `staging_enabled()` |
| 用"输出正确"证明等待有效 | 必须故意发布**陈旧** stamp 看它挂死（M0 的陈旧目标对照，M4 复用：200 s 被杀才算过） |

## 现在能跑什么

```bash
# 门禁（frac 0 / 0.55 都应与冻结 oracle 逐字节相同）
cd ~/code/naiwa-rs
BIN=$HOME/naiwa-rs/target/release/naiwa-rs TAG=X FRAC=0.55 NAIWA_NO_LOOKAHEAD=1 \
  OUT=$HOME/bench-rel/naiwa-v2/m4 timeout 400 ./harness/m1/capture_oracle.sh
diff ~/bench-rel/naiwa-v2/m4/reference.X.txt harness/m1/reference.oracle.txt   # 必须 IDENTICAL

# 结构计数（48 gate 捕获、384=48×8 replay、0 退化、ring 冻结）
NAIWA_NO_LOOKAHEAD=1 $BIN bench -m ~/models/qwen3-coder/requant \
  -f '<shard1>;<shard2>' --cpu-moe --expert-cache-gb 2 --pcie-frac 0.55 \
  --prompt-len 512 --gen-len 8 --depth 4 --iterations 1 --warmup 0

cargo test --offline --release --features cuda          # 33/33
```

## 下一步（M5c / M6）

1. **M6**：Q2 已定案（见下）——`SYNC_PLE` 作为第四个同步点（`WAITS_PER_LAYER = 4`, `K = 192`）
   + host 侧纯函数算行号 + gather DMA + 整步图；判据 = 整步 1 次 launch + 逐 token == eager。
2. **M5c（GPU predict）已降级**：M5b 之后预测线程有余量（job ~0.5 ms vs 1.6 ms 的层窗口，命中率
   已回到池基线），把 gemv 搬 GPU 换不来吞吐。等 M6 把 host 工作搬走、层窗口变窄之后再考虑——它
   能删掉整条激活通道，仍值得做，但排在 M6 之后。理由见 `worklogs/m5.md`。
3. **旧账**：一次未复现的 `frac 0.55 + NO_LOOKAHEAD` 偏离（当前构建 11/11 干净），M6 前建议
   `×50` 再逼一次。

## 开着的

- ~~**Q2（PLE 策略）**~~ **已定案**：`ple.rs` 的 `PleHash::rows` 是**纯函数**——只看最近
  `ngram-1` 个 token id(host 本来就有)+ 素数取模,没有 conv/隐状态需要在 host 复现。所以 PLE 在
  v2 里不是新机制,而是**第三个 host 产出的 DMA**:host 算 hashed 行号 → 在 copy stream 发 gather
  拷贝 → 单调发布 `SYNC_PLE(layer, pass)` → 图内自旋等它。
- **PLE 权重只在 CPU,永不进显存**(定案):fork 里的 `make_table_resident`/`ResidentTable`
  (整表搬显存,按 `RESIDENT_RESERVE_BYTES` 预算)在 v2 不用 → 12 GB 全部留给权重/专家缓存/暂存环,
  显存预算与卡大小无关;GPU 只收本 pass 的 gather 行。同一决定让 fork **再瘦 ~300 行**。
- fork 的 W 时代探针（`PLE_DBG`/`MOE_DBG`/`MTP_DBG`，~150 行）可删——hygiene，不急。
