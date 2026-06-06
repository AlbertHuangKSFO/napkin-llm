[[042 chunked prefill：切块融合|chunked prefill]] 是 [[LLM/106 chunked prefill 与 prefill、decode 解耦|chunked prefill]] 的系统视角:把一个长 [[013 Prefill 阶段：计算受限|prefill]] **切成若干小块**,每一步只放一块 prefill chunk,用剩余的 token 预算让在飞的 [[014 Decode 阶段：访存受限|decode]] 顺带 piggyback 进同一 batch。这样每步处理的 token 数被 `max_num_batched_tokens`(token budget)钳住,**单步耗时有上界**,在飞 decode 不再被长 prompt 整段卡死(stall-free),从而同时平滑 TTFT 与 TPOT/ITL。这是 Sarathi-Serve(Agrawal 2024)的核心,也是 vLLM V1 的默认调度。它和 [[048 为何分离 prefill 与 decode|PD 分离]]是治理 [[043 prefill、decode 干扰与 stall|PD 干扰]]的两条路线。

## 直觉

长 prefill 像**一锅必须连续炖 3 小时的大菜**。如果厨师(GPU)一旦开炖就不能离锅,那这 3 小时里其他桌的小炒(decode)全部停摆——客人的菜一口气上不来,体验崩了。

chunked prefill 的做法是:把这锅大菜拆成 6 段、每段炖 30 分钟,**每炖完一段就抽空去翻炒几个小炒**。大菜总时长没变(甚至略增),但其他桌的菜从此**每隔 30 分钟稳定上一道**,不再被大菜独占。

关键:GPU 一步能处理的 token 数是有预算的(比如 2048)。一个 chunk 占掉 512 个 prefill token,**剩下 1536 个预算正好让一大批 decode 搭便车**——prefill chunk 已经把算力吃满,decode piggyback 几乎不增量搬权重,等于"白嫖"了这一步的算力。

**生活类比**:GPU 像一个安检口,一步能过的 token 数有固定预算(比如 2048)。长 prompt 是件**超大行李**(4096 token)。不切块时,它一口气独占安检口约 200ms,这 200ms 里后面 60 个小包(在飞 decode)**一个都过不去**,小包的 ITL 从 40ms 飙到 200ms,5 倍尖刺,安检口被堵死。chunked prefill 的做法:把大行李拆成 8 段 × 512,**每过完一段就插空让一批小包也过**——一段 chunk 占 512 预算,余下 1536 预算正好塞满一批小包。于是单步被预算钳在 ~30ms 有上界,小包每步稳定通过、ITL 平滑无尖刺。几乎不损吞吐,因为 chunk 已把算力吃满、小包搭便车不额外搬权重;代价只是这件大行李要 8 步才过完,TTFT 略增 +10–20%。这条「单步上界」滑块就是 `max_num_batched_tokens`。

![[sched-042类比安检分段插空.svg]]

## 例子

一个 4096 token 的长 prompt,与 60 个在飞 decode 请求共存。token budget = 2048,chunk 大小 = 512。

**不切块(prefill 优先)**:长 prefill 独占一步,4096 token 的大 GEMM ≈ 200 ms。这 200 ms 里 60 个在飞请求**一个 token 都出不来** → ITL 从 40 ms 飙到 200 ms,**5× 尖刺**。

**切块**:prefill 拆成 8 块 × 512:
- 每步 = 1 个 chunk(512 prefill token) + 余下 1536 预算装满 decode。
- 单步耗时 ≈ 切块前的 1/8 量级 + decode,被预算钳住,比如 ~30 ms。
- 60 个在飞请求每步照常各出 1 token,**ITL 稳定在 ~30–40 ms,无尖刺**。
- 代价:这条 prompt 的 TTFT 略增(8 步才填完 KV vs 1 步),约 +10–20%;调度开销略增。

吞吐基本不掉:Sarathi-Serve 报告 Mistral-7B(单 A100)serving capacity **×2.6**、Falcon-180B(流水线并行)**×5.6**——因为 stall-free 让系统能开更大 batch 而不破延迟 SLO。

## 原理

设 token budget 为 $C$,某步装 1 个长度为 $c$ 的 prefill chunk 与 $d$ 个 decode($c+d \le C$)。单步耗时近似:

$$
t_\text{step} \approx \underbrace{\alpha\,(c+d)}_{\text{compute, prefill chunk 主导}} \;+\; \underbrace{\beta}_{\text{固定开销}},\qquad c+d \le C
$$

因为 $c+d \le C$ 被钳住,**$t_\text{step}$ 有上界** $\approx \alpha C + \beta$,这就是 stall-free 的数学保证:在飞 decode 的 ITL ≤ 这个上界。

chunk 大小 $c$ 是核心旋钮,体现 TTFT–TPOT 取舍:

$$
\text{TTFT}\ \propto\ \frac{S}{c}\cdot t_\text{step}\ (\text{需 } \lceil S/c\rceil \text{ 步填完}),\qquad \text{TPOT 抖动}\ \propto\ c
$$

$c$ 越小 → 单步越短、TPOT 越平稳,但填完整段 prefill 的步数越多 → TTFT 越长、调度开销越大;$c$ 越大 → TTFT 短但接近不切块的 stall。实践中 $c$ 取几百到 1–2k token,使 prefill chunk 恰好把算力推到接近饱和(算术强度过转折点,见 [[016 batch 如何改变算术强度|batch 改变算术强度]])又不至于造成可感尖刺。

## 图

![[sched-chunked-prefill切块.svg]]

## 代码

调度时把超预算的 prefill 自动切块,并让 decode piggyback:

```python
def schedule_step(running_decodes, waiting_prefills, token_budget):
    batch = []
    # ✅ decode 优先占预算（每个 1 token，关乎在飞请求 SLO）
    for req in running_decodes:
        if token_budget >= 1:
            batch.append(req.as_decode()); token_budget -= 1

    # ✅ 用剩余预算放 prefill；超预算就只取一块，下一步接着切
    for req in waiting_prefills:
        if token_budget <= 0:
            break
        remaining = req.prompt_len - req.prefilled
        chunk = min(remaining, token_budget)        # 关键：切块到正好填满预算
        batch.append(req.as_prefill_chunk(req.prefilled, chunk))
        req.prefilled += chunk                      # 记录进度，跨步续切
        token_budget -= chunk
    return batch
```

vLLM 配置:

```python
# ✅ 推荐：开启 chunked prefill，用 token 预算控制单步上界（V1 默认即开启）
llm = LLM(model="...",
          enable_chunked_prefill=True,
          max_num_batched_tokens=2048)   # 单步 token 上界 = TPOT 上界旋钮

# ❌ 反例：关掉 chunked prefill 又把 max_num_batched_tokens 调到很大
#    长 prompt 会独占一步、整段阻塞在飞 decode → ITL 周期性尖刺、TPOT SLO 破
llm = LLM(model="...", enable_chunked_prefill=False, max_num_batched_tokens=32768)
```

`❌` 关闭切块 + 巨大预算 = 长 prefill 独占步 = generation stall;`✅` 开启切块并用 `max_num_batched_tokens` 把单步钳小,换来平稳 TPOT。注意:`max_num_batched_tokens` 太小会损吞吐(prefill 利用不满算力),太大则 stall 抬头——它就是这条取舍线上的滑块。


![[sched-042chunk大小取舍.svg]]

## 面试高频

- **Q:chunked prefill 解决什么问题?** 长 prefill 独占一步会让同批在飞 decode 集体 stall(ITL 尖刺);切块 + token 预算让单步耗时有上界,decode piggyback,stall-free。
- **Q:为什么切块几乎不损吞吐?** prefill chunk 已把 GPU 算力吃满,decode 搭便车几乎不增量搬权重;而 stall-free 反而允许开更大 batch,净效果是 serving capacity 提升(Sarathi-Serve 报 2.6–5.6×)。
- **Q:chunk 大小怎么定,取舍是什么?** chunk 越小 TPOT 越平稳但 TTFT 越长、调度开销越大;越大 TTFT 短但 stall 抬头。由 `max_num_batched_tokens` 控制,取到 prefill 恰好饱和算力的点。
- **Q:chunked prefill 和 PD 分离什么关系?** 两者都治 [[043 prefill、decode 干扰与 stall|PD 干扰]]:chunked prefill 在**单实例内**用切块融合;[[048 为何分离 prefill 与 decode|PD 分离]]把 prefill/decode 拆到**不同实例**物理隔离。前者简单、后者隔离彻底但需传 KV。

## 关键事实

- **chunked prefill / stall-free batching** 出自 Sarathi-Serve(Agrawal et al., OSDI 2024);前身 SARATHI(2023)提出 chunked-prefill + decode-maximal batching(prefill chunk 饱和算力、decode piggyback)。
- 报告收益:Mistral-7B 单 A100 serving capacity **×2.6**,Yi-34B 双 A100 **×3.7**,Falcon-180B(流水线并行)端到端 **×5.6**。
- vLLM **V1(2025)** 默认开启 chunked prefill,且调度上**优先 decode、再放 prefill、放不下自动切块**;`max_num_batched_tokens` 是控制单步上界(即 TTFT–TPOT 取舍)的核心旋钮。
