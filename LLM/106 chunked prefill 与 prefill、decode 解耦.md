[[106 chunked prefill 与 prefill、decode 解耦]]:推理的 prefill(处理整段 prompt,**算力受限**)与 decode(逐 token 生成,**访存受限**)资源诉求相反、相互干扰;**chunked prefill** 把长 prefill 切块塞进 decode 批共跑(同机融合),**PD 分离**则把两阶段拆到不同 GPU 池(分机部署)——都是为了让算力与带宽同时压满、并消除长 prefill 卡住 decode 的抖动。

## ① 直觉:两阶段一个吃算力、一个吃带宽,凑一起才划算

回忆 [[078 推理算力、吞吐与延迟、Roofline|推理两阶段]]:

- **Prefill**:一次性读完整个 prompt,并行算出所有 token 的 K、V。一次**大矩阵乘(GEMM)**,计算量大——**算力受限**,带宽闲着。决定 TTFT(首 token 时延)。
- **Decode**:逐个 token 自回归生成,每步只算 1 个 token,却要把整个模型权重读一遍——**访存受限**,算力闲着。决定 TPOT(每 token 时延)。

天真做法是「先把一条请求 prefill 完,再 decode」,但同机服务多请求时出大问题:**一条新来的长 prompt 要 prefill,会霸占整个 GPU 算力一整步,把所有正在 decode 的老用户全卡住**——他们这一步出不了 token,体验上是「卡顿(generation stall)」,TPOT 剧烈抖动。

两个解法,对应两种部署哲学:

- **chunked prefill(同机融合)**:别让长 prefill 独占一步。把它**切成小块**,每步只跑一块,**和正在 decode 的请求拼进同一个 batch**。prefill 块吃满算力,decode「搭车(piggyback)」用闲置的带宽——两者性质互补,几乎免费叠加。
- **PD 分离(prefill/decode disaggregation)**(系统设计深入见 [[LLM Infra/048 为何分离 prefill 与 decode|PD 分离]]):干脆把 prefill 和 decode **拆到两组不同的 GPU 上**,各配最划算的硬件、各自扩缩、互不干扰;代价是 prefill 算完要把 KV-Cache **传**给 decode 机。

## ② 例子:4096 token 的 prompt,切 vs 不切

设 prompt 4096 token,decode 块大小上限设 512。

**不切块**:这条 prefill 一步算完 4096 token,这一步耗时长(约 8 倍于一个 decode 步)。期间批里其他 N 个用户的 decode 全部停摆——他们的 TPOT 这一步 spike 到 8 倍。TTFT 倒是快(一步出首 token)。

**chunked prefill(块=512)**:4096 切成 8 块。每步跑 512 个 prefill token + 同时跑批里的 decode(比如 30 个)。

- 每步 prefill 块(512 token 的 GEMM)已足够吃满算力;decode 这 30 个 token 利用该步没用满的访存带宽,**近乎免费**地搭车完成。
- 老用户的 decode **不再被卡 8 步,而是每步都正常出 token**,TPOT 平稳。
- 代价:这条 prefill 的 TTFT 从「1 步」变成「8 步」,**首 token 略慢**,但抖动消失、整体更稳。

**块大小是旋钮**:块越小,decode 卡顿越轻(TPOT 越稳),但 prefill 拖得越长(TTFT 升)且每块开销摊得多;块越大反之。生产上按 SLA 调。

**TTFT/TPOT 量化(把抖动算清)。** 设一个纯 decode 步耗时 $t$,4096 token 的整块 prefill 约 $8t$(算力受限,正比于 token 数)。
- **不切块**:这条请求 TTFT $\approx 8t$(一步出首 token,快);但批里其他 30 个用户这一步的 TPOT 从 $t$ **spike 到 $8t$**(被独占),体验上是明显卡顿。
- **chunked(块=512,切 8 块)**:每步跑 512 prefill + 30 decode,每步耗时仍约 $t$(512 的 prefill GEMM 刚好吃满算力、decode 搭车几乎不加时)。这条请求 TTFT 变成 $8t$(8 步才 prefill 完,**首 token 略慢**),但其他用户 TPOT **全程稳定在 $t$**,无 spike。

**这就是 chunked prefill 的本质交换**:牺牲少量 TTFT(首 token 稍慢)换 TPOT 全程平稳 + 整体吞吐提升。对「重交互、低抖动」的 chat SLA 极划算。块大小决定这个折中点:块=128 时 TPOT 几乎零抖但 TTFT 升、调度开销大;块=2048 时 TTFT 好但偶有小 spike。

![[infer-chunked-prefill.png]]

## ③ 原理:piggybacking、decode-maximal batching 与 PD 分离

**chunked prefill 的核心(Sarathi, 2023)。** 把 prefill 请求切成等大的 chunk;构造 batch 时用「**一个 prefill 块 + 尽量多的 decode 请求**」(decode-maximal batching)。为什么有效?

- **prefill 块算力受限、带宽闲**;**decode 访存受限、算力闲**。两者拼一步:prefill 把计算单元喂满,decode 把闲置的访存带宽喂满——**同一步里算力和带宽双双压满**,decode 的边际成本极低(piggyback,搭便车)。
- 解决了两件事:① 长 prefill 不再独占一步、不再卡住 decode → **TPOT 稳定**;② GPU 不再在「纯 decode 步」浪费算力 → **吞吐提升**。

这与 [[105 连续批处理 continuous batching|连续批处理]] 天然契合:连续批每步要拼「老请求 decode + 新请求 prefill」,chunked prefill 正是让新请求的长 prefill 不破坏这一步节奏的手段。vLLM、SGLang 默认开启。

**用 Roofline 看「搭车」为何近乎免费(讲透物理)。** 一个 GPU 步的耗时 = max(计算时间, 访存时间)(见 [[078 推理算力、吞吐与延迟、Roofline|Roofline]])。
- **纯 decode 步**:访存时间长(读全部权重)、计算时间短(只算几十个 token)→ 步耗时被**访存**卡住,**算力大量闲置**。
- **纯 prefill 步(一块 512 token)**:计算时间长(大 GEMM)、访存时间相对短 → 步耗时被**计算**卡住,**访存有余**。
- **两者拼一步**:计算时间 ≈ prefill 块的(吃满算力),访存时间 ≈ 读权重的(本来就要读)。decode 那几十个 token 的额外计算**塞进了 prefill 没占满的访存窗口**,几乎不增加步耗时。这就是「decode 免费搭 prefill 的车」的物理本质:**一个吃计算、一个吃访存,正好互补填满 Roofline 的两个轴**。

![[infer-PD搭车Roofline.png]]

**token budget 怎么定(实现旋钮)。** 每步给一个总 token 预算(如 512),先放尽量多的 decode(每个 1 token),剩余预算给一个 prefill 块。预算大 → 单步能塞更多 prefill、TTFT 好但 decode 可能被挤;预算小 → decode 优先、TPOT 稳但 prefill 慢。vLLM 的 `max_num_batched_tokens` 就是这个旋钮。

**PD 分离(disaggregation)的核心(DistServe / Mooncake, 2024)。** 同机融合再优,prefill 和 decode 仍**共享同一份硬件和并行配置**,但它们诉求相反:prefill 想要**高算力**(配 H100、FP8、TP 切分),decode 想要**高带宽 + 大 batch**(配高带宽卡、更激进的批)。分离后:

- **两个独立 GPU 池**:prefill 池只做 prefill、按 TTFT 扩缩;decode 池只做 decode、按吞吐扩缩。
- prefill 池算完一条的 KV-Cache,**通过 NVLink / RDMA 传给 decode 池**继续生成。
- 好处:彻底消除两阶段干扰、各池用最划算的硬件和并行策略、独立弹性伸缩。代价:**跨机传 KV 的带宽与延迟**(长 prompt 的 KV 可能很大),需要高速互联和精心调度(传输与计算重叠)。

**为什么 PD 分离能各自配最优硬件/并行(讲透)。** prefill 是大 GEMM、算力受限,适合**张量并行 TP**(切大矩阵、吃满 TensorCore)、用算力强的卡;decode 是访存受限、要大 batch,适合**更激进的批 + 高带宽**,甚至不同的并行度。同机融合时两阶段被迫共享一套 TP/PP 配置,只能折中;分离后各取所需。另一个收益:**两池独立弹性伸缩**——白天交互多(decode 重)就扩 decode 池,批处理任务多(prefill 重)就扩 prefill 池,而不是整体扩容。

**PD 分离的代价与门槛。** ① **跨机传 KV**:一条 32K prompt 的 KV 可能几 GB,要在 prefill 算完后通过 NVLink/RDMA 传给 decode 机,传输与计算需重叠(否则 decode 机干等);长 prompt 时传输可能成新瓶颈。② **运维复杂**:两套池、两套扩缩、KV 路由与一致性,工程量远大于同机。③ **小规模不划算**:请求量不大时,跨机传输的固定开销 + 两池利用率不满,反而比单机融合差。所以**中小规模用 chunked prefill,超大规模 + 强 SLA 才上 PD 分离**。

**两者关系**:chunked prefill 是「在一台机器内把两阶段融合得更顺」,PD 分离是「把两阶段彻底拆开各自优化」。前者简单省机器、适合中小规模;后者复杂但在大规模、强 SLA(同时要低 TTFT 和低 TPOT)下收益大。可叠加:分离后 decode 池内部仍用连续批 + chunked prefill 处理零散 prefill。

![[infer-PD分离部署.png]]

## ④ 代码:chunked prefill 调度(伪代码)

```python
def build_batch(running_decodes, prefill_queue, chunk_size=512, token_budget=512):
    """decode-maximal batching:一个 prefill 块 + 尽量多的 decode。"""
    batch, used = [], 0

    # ① 先放一个 prefill 块(切块,不让长 prompt 独占)
    if prefill_queue:
        req = prefill_queue[0]
        chunk = req.next_chunk(chunk_size)        # 取本请求下一个 512 token 块
        batch.append(("prefill", req, chunk))
        used += len(chunk)
        if req.prefill_done():                    # 切完最后一块,转入 decode 队列
            prefill_queue.pop(0)

    # ② 用剩余预算塞 decode(每个只 1 token,几乎免费搭车)
    for req in running_decodes:
        if used >= token_budget:                  # 也可让 decode 不受预算限,见实现
            break
        batch.append(("decode", req, req.last_token))
        used += 1
    return batch

# ❌ 错:长 prefill 独占一步
#    run_prefill(req_4096)          # 这一步 ~8× decode 耗时,卡死所有 decode 用户
# ✅ 对:切块 + 与 decode 拼批,每步算力+带宽双满,TPOT 平稳
#    for chunk in split(req_4096, 512):
#        run_batch([("prefill", req, chunk)] + current_decodes)
```

## 面试高频

- **Q:prefill 和 decode 分别受什么限制?为什么要解耦?** prefill 算力受限(大 GEMM)、decode 访存受限(逐 token 读权重);诉求相反且长 prefill 会卡住 decode,故需融合(chunked prefill)或分离(PD 分离)来同时压满算力与带宽、消除干扰。
- **Q:chunked prefill 怎么提速、怎么稳 TPOT?** 把长 prefill 切块,每步「一块 prefill + 多个 decode」拼批:prefill 吃算力、decode 用闲置带宽搭车;长 prefill 不再独占一步,decode 每步正常出 token,TPOT 平稳、吞吐升。
- **Q:chunked prefill 的块大小怎么权衡?** 块小→decode 卡顿轻、TPOT 稳,但 TTFT 升、开销摊多;块大→TTFT 快但易卡 decode。按 SLA 调。
- **Q:PD 分离相比 chunked prefill 的优劣?** 分离彻底隔离两阶段干扰、各配最优硬件与并行、独立扩缩,适合大规模强 SLA;代价是跨机传 KV(带宽/延迟)。chunked prefill 同机更简单省钱。
- **Q:为什么 decode 能「免费」搭 prefill 的车?** prefill 步算力被吃满但带宽闲,decode 恰好访存受限、只需带宽,塞进同一步几乎不增加耗时。
- **Q:PD 分离要传什么、走什么通道?** 传 prefill 算好的 KV-Cache,经 NVLink/RDMA;长 prompt 的 KV 大,需高速互联 + 传输与计算重叠。
- **Q:chunked prefill 对 TTFT 是好是坏?** 略坏——长 prefill 被切成多步,首 token 来得稍慢;但换来 TPOT 全程平稳(无 spike)和整体吞吐提升。对低抖动 SLA 划算。
- **Q:什么规模该上 PD 分离?** 超大规模 + 强 SLA(同时要低 TTFT 与低 TPOT)。中小规模用 chunked prefill 更省:PD 分离有跨机传 KV、两池运维、利用率不满等固定开销,小流量不划算。
- **Q:PD 分离为什么能各配最优硬件?** prefill 算力受限(适合强算力卡 + TP),decode 访存受限(适合高带宽 + 大 batch),同机被迫共享一套配置只能折中;分离后各取所需,且两池独立弹性伸缩。

## 关键事实

- Prefill 算力受限、decode 访存受限,是两阶段解耦的物理依据(《LLM Inference Unveiled》, Yuan et al., 2024, arXiv:2402.16363,见 [[078 推理算力、吞吐与延迟、Roofline]])。
- chunked prefill + decode-maximal batching:把 prefill 切等大块、每批一个 prefill 块搭载尽量多的 decode,prefill 饱和算力、decode 搭车,提吞吐并稳 TPOT(Sarathi, Agrawal et al., 2023, arXiv:2308.16369;Sarathi-Serve, OSDI 2024)。
- chunked prefill 与连续批处理天然契合,vLLM、SGLang 默认启用,用以避免长 prompt 阻塞 decode(vLLM 文档,2024,见 [[105 连续批处理 continuous batching]])。
- PD 分离(prefill/decode disaggregation):两阶段拆到独立 GPU 池,各自优化硬件与并行、独立扩缩,通过高速互联传 KV-Cache,显著改善 TTFT/TPOT 的同时满足(DistServe, Zhong et al., 2024, arXiv:2401.09670;Mooncake, Moonshot AI, 2024)。
- 块大小、传输带宽是关键旋钮:chunked prefill 块大小权衡 TTFT 与 TPOT;PD 分离受跨机 KV 传输带宽/延迟约束。
- chunked prefill 的折中:牺牲少量 TTFT(首 token 因切块略慢)换 TPOT 全程平稳无 spike 与吞吐提升;块小则 TPOT 更稳但 TTFT 升、调度开销大,块大反之。
- PD 分离让两阶段各配最优硬件与并行(prefill 强算力+TP、decode 高带宽+大 batch)、独立弹性伸缩;代价是跨机传 KV(长 prompt 几 GB)、运维复杂、小规模不划算。
- Sarathi-Serve(OSDI 2024)在保证 SLA 的同时用 chunked prefill 平衡 TTFT/TPOT,是该思想的系统化落地;Mooncake(Moonshot)将 PD 分离 + 全局 KV 缓存池用于生产级长上下文服务。
