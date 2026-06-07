[[044 调度器设计：waiting、running 队列与抢占|调度器设计]]把 [[041 连续批处理：迭代级调度内幕|迭代级调度]]落成具体的数据结构与策略。一个 LLM 推理调度器(以 vLLM 为范本)维护两个队列:**waiting**(到达但还没拿到 [[LLM/102 KV-Cache|KV-Cache]] 块的请求)和 **running**(已准入、每步前进一个 token 的请求)。每一步,调度器做**准入判定**——按 **KV 显存预算**和 **token 预算**决定从 waiting 拉多少进 running;当 running 的 KV 占用透支显存时,触发**抢占**(preemption)把请求踢回去腾块。默认排序 **FCFS**(先到先服务)。这套"两队列 + 预算准入 + 抢占"就是连续批处理的调度骨架,抢占细节见 [[045 抢占：重计算 vs swap|重算 vs swap]],优先级与公平见 [[046 优先级、公平性与饥饿|优先级与饥饿]]。

## 直觉

把调度器想成一家**没有等位区上限的餐厅经理**,但厨房(显存)的灶台(KV 块)是有限的:

- **waiting = 门口排队的客人**:已登记但还没入座(还没分到灶台资源)。
- **running = 已入座正在用餐的客人**:每"一轮上菜"(一步)都吃一口(出一个 token)。
- **准入**:每轮上菜前,经理看灶台还空几个 → 从门口叫人入座;但得先估这桌要点多大的菜(prompt + 预计输出的 KV),灶台不够就不叫。
- **抢占**:万一在座客人点的菜越来越多(输出越来越长、KV 涨),灶台不够用了,经理只能**请某桌暂时离席**(踢回 waiting),把灶台让出来,稍后再请回来续上。

核心约束:**running 里所有人的 KV 必须同时装进显存**。装不下,就只能"少准入 + 抢占"——这是显存墙(见 [[010 显存墙与 LLM 推理的本质约束|显存墙]])在调度层的直接体现。

## 例子

显存可容纳 100 个 KV 块,token budget = 2048。某步状态:

- running = {R1 用 30 块, R2 用 25 块, R3 用 20 块} → 已用 75 块,空闲 25 块。
- waiting = [R4(prefill 需 30 块), R5(需 10 块)],FCFS。

**准入判定**:
- R4 需 30 块 > 空闲 25 块 → **R4 准入失败**(也未触发切块的话就排队等)。
- R5 需 10 块 ≤ 25 块,且其 prompt token ≤ 剩余 token 预算 → **R5 准入**,移入 running。现在用 85 块。

**几步之后**:R1/R2/R3 各自 decode,KV 随输出增长,某步总需求涨到 105 块 > 100。**透支!触发抢占**:
- 按策略(常 LIFO 或低优先级先)挑 R3 抢占 → 释放它的 20 块 → 回到 85 块,可继续。
- R3 被踢回 waiting 队头,稍后用 [[045 抢占：重计算 vs swap|recompute 或 swap]]恢复。

这就是稳态下调度器的日常:**在显存边界上反复做准入与抢占的平衡**。

## 原理

设显存可用 KV 块 $M$,running 集合 $R$,请求 $i$ 当前 KV 块数 $b_i$(随其序列长增长)。可行性不变式:

$$
\sum_{i\in R} b_i(t)\ \le\ M,\qquad \sum_{i\in R} n_i(t)\ \le\ C\ (\text{token budget})
$$

**准入**:从 waiting 队头(FCFS)取请求 $j$,当且仅当

$$
\sum_{i\in R} b_i + b_j \le M\quad\text{且}\quad \sum_{i\in R} n_i + n_j \le C
$$

**抢占触发**:decode 使某些 $b_i$ 增长,若某步 $\sum b_i > M$,需选victim集合 $P\subseteq R$ 使

$$
\sum_{i\in R\setminus P} b_i \le M
$$

并对 $P$ 中请求执行 recompute / swap,踢回 waiting。victim 选择策略影响公平与吞吐(见 [[046 优先级、公平性与饥饿|饥饿]]),vLLM 默认近似 LIFO(后进先被抢,保护老请求避免饿死)。

**为什么 KV 预算是主约束而非 token 预算?** 因为 decode 是 [[014 Decode 阶段：访存受限|访存受限]]的,瓶颈在显存装不装得下 KV,而非算力;token budget 主要钳 prefill 的单步大小(防 [[043 prefill、decode 干扰与 stall|stall]])。

## 图

![[sched-调度器状态机.png]]

## 代码

调度器骨架(两队列 + 预算准入 + 抢占):

```python
class Scheduler:
    def __init__(self, kv_blocks, token_budget):
        self.waiting = deque()      # FCFS
        self.running = []
        self.free_blocks = kv_blocks
        self.token_budget = token_budget

    def schedule(self):
        batch = []
        # 1) running 的 decode 先排（每个 1 token，必须保证它们的 KV 能扩张）
        for req in self.running:
            need = req.next_block_demand()          # 这步可能要新增 1 个 KV 块
            while self.free_blocks < need:          # ✅ 显存不够 → 抢占腾块
                victim = self.pick_victim()         # LIFO / 低优先级先
                self.free_blocks += victim.kv_blocks
                self.preempt(victim)                # recompute or swap，踢回 waiting
            self.free_blocks -= need
            batch.append(req.as_decode()); self.token_budget -= 1

        # 2) 用剩余预算 + 空闲块从 waiting 准入新请求
        while self.waiting and self.token_budget > 0:
            req = self.waiting[0]
            if req.kv_demand() <= self.free_blocks:     # ✅ KV 装得下才准入
                self.waiting.popleft()
                self.free_blocks -= req.kv_demand()
                chunk = min(req.prompt_len, self.token_budget)
                batch.append(req.as_prefill(chunk)); self.token_budget -= chunk
                self.running.append(req)
            else:
                break                                    # 队头放不下，FCFS 不跳过
        return batch

    # ❌ 反模式：准入时不查 KV 预算，盲目塞满 running
    #    → 某步 KV 透支，要么 OOM 崩溃，要么被迫大规模抢占、吞吐雪崩
```

`❌` 不做 KV 预算检查的准入会导致显存透支(OOM 或抢占风暴);`✅` 准入和 decode 扩张都必须先确认 KV 块够,不够就抢占。注意 FCFS 下队头放不下时**不跳过**后面的(否则破坏公平、易饿死长请求),但配优先级时可调(见 [[046 优先级、公平性与饥饿|优先级]])。


![[sched-044准入抢占判定.png]]

![[sched-044KV预算占用墙.png]]

## 面试高频

- **Q:推理调度器维护哪两个队列,各装什么?** waiting(已到达、未拿到 KV 块)和 running(已准入、每步前进一 token);调度就是按预算在两者间搬请求。
- **Q:准入的硬约束是什么?** 两个预算:KV 显存块够装(主约束,因 decode 访存受限)、单步 token 数 ≤ token budget(钳 prefill 防 stall)。
- **Q:什么时候触发抢占?** running 请求 decode 时 KV 持续增长,某步总 KV 需求超过显存可用块,必须踢出 victim 腾块。
- **Q:victim 怎么选?为什么不简单选最大的?** vLLM 默认近似 LIFO(后进先抢),保护早到的老请求避免饿死、避免做过的工作白费;纯按大小选会反复抢占长请求造成饥饿(见 [[046 优先级、公平性与饥饿|饥饿]])。
- **Q:为什么是 KV 而不是算力成为调度主约束?** decode 访存受限,瓶颈在显存能否装下所有在飞请求的 KV-Cache,而非单步算力。

## 关键事实

- vLLM 调度器维护 **waiting / running**(及被换出的 swapped)队列,默认调度策略 **FCFS**,准入受 **KV 块预算**与 **`max_num_batched_tokens`** 双重约束。
- KV 显存是连续批的**主约束**:running 批所有请求的 KV-Cache 必须同时装进显存,装不下即触发 [[045 抢占：重计算 vs swap|抢占]];PagedAttention 用分页块管理 KV 以减少碎片。
- 截至 2025,vLLM **V1** 调度先排 decode、再放 prefill(放不下切块),抢占默认 **recompute**;`max_num_seqs` 控制 running 批最大请求数,`max_num_batched_tokens` 控制单步 token 上界。
