[[105 连续批处理 continuous batching]]:把推理调度的粒度从「一整批请求」降到「每一步迭代」(iteration-level scheduling)——某条请求一生成完就立刻让出槽位、新请求即刻补入,GPU 不再空转等最长那条,吞吐相比静态批处理提升数倍。由 Orca(2022)提出,是 vLLM 等现代引擎的吞吐基石。

## ① 直觉:静态批的「木桶效应」——全批被最长那条拖死

[[078 推理算力、吞吐与延迟、Roofline|decode 访存受限]],权重每步要整读一遍,所以**攒 batch** 是提吞吐的关键:一批 token 共摊一次权重读取。但天真地「攒一批 → 一起跑到全完 → 再换下一批」(**静态批处理 / request-level batching**)有个致命浪费:

**批里每条请求生成长度不同,但必须对齐到最长那条**。短请求早早生成完了,却只能**占着槽位空转**,陪着最长那条跑到底。GPU 在算一堆「已经结束」的空槽——利用率低,而排队的新请求还得等整批结束才能进。这就是木桶效应:批的速度被最长那条决定。

连续批处理(continuous batching,又叫 in-flight batching、dynamic batching)的招数:**别按「批」调度,按「步」调度**。每生成一步后就检查——谁结束了?把它踢出、立刻让一个等待中的新请求补进它的槽位。于是 batch 是「流动」的:槽位始终满载,短请求不拖累长请求,新请求随到随进。

## ② 例子:4 槽,短请求让位给新请求

设 GPU 同时跑 4 条,各自生成长度:R1=3、R2=5、R3=2、R4=8 步。

**静态批处理**:整批必须跑满 8 步(R4 最长)。R3 在第 2 步就完了,却空转 6 步;R1 空转 5 步;R2 空转 3 步。**4×8=32 个槽位·步中,只有 3+5+2+8=18 个在真正干活,利用率 18/32 ≈ 56%**。期间到达的新请求只能干等下一批。

**连续批处理**:R3 第 2 步完成 → 第 3 步它的槽位立刻被新请求 R5 接管;R1 第 3 步完成 → 第 4 步换 R6……每个空出的槽位即刻被填。**利用率逼近 100%**,同样时间内服务的请求数显著更多。实测吞吐相比静态批可提升数倍(Orca/vLLM 量级)。

代价/权衡:它不直接降单请求延迟(每条还是逐 token 出),但因为不用排队等批、GPU 不空转,**平均排队时延和整体吞吐**大幅改善。

**再算一笔吞吐账。** 设 decode 一步(不论批里多少条)耗时约 $t$(因为访存受限,主要时间是把权重读一遍,与 batch 大小近似无关)。
- 静态批:跑完这批要 $8t$(对齐最长),期间真正产出 $18$ 个 token → 吞吐 $\frac{18}{8t}=\frac{2.25}{t}$ tok/s。
- 连续批:槽位始终满载 4 条,每 $t$ 产出 $4$ 个 token → 吞吐 $\frac{4}{t}$ tok/s。

**约 1.78× 提升**——而且这还是「短长差距不大」的温和例子;真实流量里请求长度从几十到几千 token 不等,木桶效应更严重,连续批相对静态批的实测吞吐提升常达 **数倍到 20+ 倍**(Orca 论文量级)。核心就一句:**decode 一步的成本几乎与 batch 无关,所以把 batch 填满 = 几乎免费地多产出 token**。

![[infer-连续批处理对比.png]]

## ③ 原理:iteration-level scheduling + 与 PagedAttention 的耦合

**核心:迭代级调度(iteration-level scheduling)。** Orca(Yu et al., OSDI 2022)指出,LLM 自回归生成天然是「一步一步」的,调度也该一步一步做。每个 decode 步(iteration)结束,调度器执行:

1. **回收**:扫描批中哪些请求生成了 EOS / 达到 max_tokens → 标记完成,**释放其 KV-Cache 与槽位**。
2. **准入**:从等待队列取新请求填补空槽,直到显存/批容量上限。
3. **拼批**:把「正在 decode 的老请求」与「刚进来要 prefill 的新请求」可能拼进同一步执行(涉及 prefill 与 decode 混跑,见 [[106 chunked prefill 与 prefill、decode 解耦|chunked prefill]])。

![[infer-迭代级调度循环.png]]

**为什么能这样拼?** 不同请求当前长度不同、KV-Cache 长度不同。要让它们在一个 batch 里跑,KV 不能再要求「整块连续预留」——否则换进换出会产生大量碎片。这正是 [[026 PagedAttention 与 KV 分页|PagedAttention]] 的用武之地:KV 按页分配,请求随到随分页、完成随时整页归还,**碎片极小、槽位复用灵活**。所以「连续批处理 + PagedAttention」是绝配:前者是调度策略,后者是支撑它的显存管理。

**新批容量受什么限制?** 不是固定的 batch size,而是**显存里还能塞下多少 KV-Cache**。请求越长、KV 越大,能并发的就越少;[[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV 压缩]]能让同样显存塞更多请求,直接放大连续批的吞吐。

**吞吐 vs 延迟。** 连续批本质还是「攒 batch 提吞吐」(见 [[078 推理算力、吞吐与延迟、Roofline|吞吐延迟]]):batch 越满、访存摊得越薄、吞吐越高;但每条请求的 TPOT 会因同批请求多而略升。生产上用「最大批 + 显存上限」平衡。

**两个生产级旋钮与陷阱。**
- **抢占与重计算(preemption)**:显存吃紧时,调度器可能要把某些正在跑的请求「踢出去」腾 KV。两种回收策略:① **swap**(把 KV 换到 CPU 内存,需要时换回);② **recompute**(直接丢弃 KV,等有空再从头 prefill)。vLLM 默认 recompute——因为 prefill 是算力受限、很快,而 swap 占 PCIe 带宽。被抢占的请求体验上是一段「卡顿」。
- **公平性与饥饿**:若一味让短请求插队,长请求可能长期排不进(饥饿)。调度器需平衡吞吐与公平,常用 FCFS + 准入控制,或给长等待请求提优先级。
- **prefill 与 decode 同批的干扰**:新请求的 prefill 是大 GEMM,塞进 decode 批会拖慢这一步的所有 decode(TPOT 抖动)——这正是 [[106 chunked prefill 与 prefill、decode 解耦|chunked prefill]] 要解决的,把长 prefill 切块,避免独占一步。

**连续批 ≠ 动态批(dynamic batching)。** 传统服务的 dynamic batching(如 Triton 早期)是「攒够一小批请求或等够时间窗就一起跑、跑完整批再换」——仍是 request-level,只是凑批时机灵活。连续批是 **iteration-level**,粒度细到「每步」,完成即让位。两者不是一回事,面试常混。

## ④ 代码:连续批调度器(伪代码)

```python
class ContinuousBatchScheduler:
    def __init__(self, model, kv_pool, max_running):
        self.model = model
        self.kv = kv_pool            # 分页 KV 池(PagedAttention)
        self.max_running = max_running
        self.waiting, self.running = [], []   # 等待队列 / 在跑请求

    def add_request(self, req):
        self.waiting.append(req)     # 新请求随到随入队

    def step(self):
        # ① 准入:用空闲槽位 + 足够 KV 页,从等待队列拉新请求
        while (len(self.running) < self.max_running
               and self.waiting and self.kv.has_free_pages()):
            req = self.waiting.pop(0)
            req.kv_blocks = self.kv.alloc(req.prompt_len)  # 按需分页
            self.running.append(req)

        # ② 拼一个 batch,跑「一步」(老请求 decode + 新请求 prefill 混跑)
        logits = self.model.forward_batch(self.running)    # 权重只读一遍
        for req, lg in zip(self.running, logits):
            req.append(sample(lg))

        # ③ 回收:本步生成完的,立刻释放 KV 与槽位
        done = [r for r in self.running if r.finished()]
        for r in done:
            self.kv.free(r.kv_blocks)                       # 整页归还,无碎片
        self.running = [r for r in self.running if not r.finished()]
        return done

# ❌ 错(静态批):for batch in batches: run_until_all_done(batch)
#    → 短请求空转等最长那条,新请求干等下一批,GPU 利用率低
# ✅ 对(连续批):每 step 回收完成的、补入新请求,槽位始终满载
```

```python
# 吞吐对比:静态批 vs 连续批(同一组请求长度,decode 一步耗时 t 与 batch 无关)
def throughput(lengths, slots, mode):
    if mode == "static":
        steps = max(lengths)                  # 对齐最长那条
        tokens = sum(lengths)                 # 真正产出的 token
        return tokens / steps                 # tok / 步
    else:  # continuous:槽位始终满载,总步数 = ceil(总token / 槽位)
        import math
        total = sum(lengths)
        steps = math.ceil(total / slots)      # 假设有源源不断的新请求补槽
        return total / steps                  # ≈ slots(满载)

L = [3, 5, 2, 8]
print("静态批:", throughput(L, 4, "static"))       # 18/8 = 2.25 tok/步
print("连续批:", throughput(L, 4, "continuous"))   # ≈ 4.0 tok/步 → ~1.78× 提升
# 真实流量请求长度差异更大(几十~几千),木桶效应更狠,实测可达数倍~20+×
```

## 面试高频

- **Q:连续批处理和静态批处理区别?** 静态批按「整批请求」调度,必须等批内最长那条跑完才换批,短请求空转、新请求排队;连续批按「每步迭代」调度,完成即让位、新请求即补入,GPU 不空转,吞吐提升数倍。
- **Q:连续批为什么能提吞吐?** decode 访存受限,权重每步只读一遍却服务满批;连续批让槽位始终满载、batch 利用率逼近 100%,摊薄访存。
- **Q:连续批和 PagedAttention 什么关系?** 互补。连续批是调度策略(谁进谁出),PagedAttention 是显存管理(KV 分页、随进随出无碎片)。后者支撑前者灵活换槽,vLLM 把两者结合。
- **Q:连续批的并发上限由什么决定?** 不是固定 batch size,而是显存能容纳的 KV-Cache 总量;请求越长越占 KV,能并发越少。KV 压缩可放大并发。
- **Q:连续批会降低单请求延迟吗?** 不直接降 TPOT,但消除排队等批 + GPU 空转,显著改善平均排队时延和整体吞吐;批越满 TPOT 略升,需平衡。
- **Q:Orca 的核心贡献是什么?** iteration-level scheduling(迭代级调度)+ selective batching,把调度粒度从请求降到迭代,奠定连续批处理。
- **Q:连续批和传统 dynamic batching 区别?** dynamic batching 仍是 request-level(凑一小批一起跑到完),连续批是 iteration-level(每步回收/补入)。粒度不同,后者才消除木桶效应。
- **Q:显存不够时连续批怎么办?** 抢占:把部分请求 KV 换出(swap 到 CPU)或丢弃(recompute 重新 prefill)。vLLM 默认 recompute——prefill 算力受限很快,比走 PCIe swap 划算。被抢占请求会卡顿。
- **Q:连续批有公平性问题吗?** 有。一味让短请求插队会让长请求饥饿;需 FCFS + 准入控制或等待时间提优先级来平衡吞吐与公平。

## 关键事实

- 连续批处理(continuous / in-flight batching)的核心是 iteration-level scheduling:每步回收完成请求、准入新请求,使 GPU 槽位持续满载,相比静态(request-level)批处理吞吐提升数倍(Orca, Yu et al., OSDI 2022)。
- 静态批处理的浪费源于批内请求生成长度不齐,须对齐到最长一条,短请求占槽空转(木桶效应)。
- 连续批与 PagedAttention 配合:KV 按页随进随出、碎片极小,使灵活换槽成为可能,是 vLLM 高吞吐的两大支柱(Kwon et al., 2023, arXiv:2309.06180,见 [[026 PagedAttention 与 KV 分页]])。
- 并发上限由显存可容纳的 KV-Cache 总量决定,而非固定 batch size;KV 压缩可提升并发(见 [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享]])。
- 连续批本质仍是「攒 batch 提吞吐」,受 decode 访存受限的物理规律支配(见 [[078 推理算力、吞吐与延迟、Roofline]]);TensorRT-LLM 称之为 in-flight batching,vLLM/SGLang 默认开启。
- 显存不足时调度器抢占:swap(KV 换出到 CPU)或 recompute(丢弃 KV 重新 prefill),vLLM 默认 recompute;需平衡公平性以免长请求饥饿。
- 连续批(iteration-level)区别于传统 dynamic batching(request-level,凑批时机灵活但仍整批跑完才换);前者粒度细到每步,是消除木桶效应的关键。
- decode 一步耗时近似与 batch 无关(访存受限,主要读权重),故「填满 batch」近乎免费提吞吐——这是连续批吞吐增益的物理根源。
