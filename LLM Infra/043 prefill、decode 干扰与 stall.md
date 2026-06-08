[[043 prefill、decode 干扰与 stall|PD 干扰]]是连续批混批的"原罪":当一个长 [[013 Prefill 阶段：计算受限|prefill]] 和一批在飞 [[014 Decode 阶段：访存受限|decode]] 被塞进同一步,prefill 的大 GEMM 独占算力、把这一步拖长,在飞 decode 在这段时间**出不了 token**——这就是 **generation stall(生成停顿)**,表现为 ITL/TPOT 的周期性尖刺。根因是 prefill(计算受限、又宽又大)与 decode(访存受限、又细又长)抢同一步的 GPU。两条治理路线:**要么 chunk**(在单实例内用 [[042 chunked prefill：切块融合|chunked prefill]] 切块钳住单步),**要么分离**(用 [[048 为何分离 prefill 与 decode|PD 分离]]把两者放到不同实例)。理解这道干扰,是理解 [[041 连续批处理：迭代级调度内幕|连续批]]为何还要继续演化的关键。

## 直觉

想象一条**单车道公路**(GPU 的一步算力),平时跑的是一串小轿车(decode,一辆接一辆稳定通过)。突然一辆**超长挂车(长 prefill)**驶入车道——它太宽太长,占满整条道,后面所有小轿车**全部刹停**,直到挂车开过去才恢复。

对司机(在飞请求的用户)来说,体验就是:本来每 40 ms 出一个字,突然卡住 200 ms 一个字不出,然后又恢复——**周期性卡顿**。挂车越长(prompt 越长),卡得越久。

为什么非要让挂车上同一条道?因为连续批要"新请求即补位",而新请求的第一步必然是 prefill。于是矛盾天生:**要补位就要混 prefill,混 prefill 就会卡 decode**。chunked prefill 把挂车拆成几节短车厢分批过,PD 分离干脆给挂车修一条专用道。

## 例子

在飞请求 B 正常 ITL = 40 ms。某步混入请求 A 的 4096-token 长 prefill(~200 ms):

| 时刻 | 事件 | B 的 ITL |
|---|---|---|
| t0–t1 | 纯 decode 步 | 40 ms ✓ |
| t1–t2 | **混入 A 的长 prefill** | **200 ms ✗(stall)** |
| t2–t3 | 恢复纯 decode | 40 ms ✓ |

- B 在这一步**少出了应得的 ~4 个 token**(200/40 ≈ 5,实际只出 1)。
- 若 SLO 要求 P99 ITL < 80 ms,这一步直接击穿——**单个长 prompt 就能让一片在飞请求集体破 SLO**。
- prompt 越长、并发越高,被波及的在飞请求越多,stall 越显眼。

这正是 Sarathi-Serve 论文的出发点:**prefill-prioritizing 调度器会因为 prefill 任意长而造成 generation stall**,使吞吐和延迟无法兼得。

## 原理

一步内若混入 prefill chunk 长度 $c$ 与 $d$ 个 decode,单步耗时由计算量主导:

$$
t_\text{step} \approx \max\!\Big(\underbrace{\tfrac{2 N_\text{params}(c+d)}{\text{FLOP/s}}}_{\text{compute}},\ \underbrace{\tfrac{\text{bytes moved}}{\text{BW}}}_{\text{memory}}\Big)
$$

纯 decode 步 $c=0$、$d$ 小,memory-bound,$t_\text{step}$ 小;一旦 $c$ 很大(整段长 prefill),compute 项暴涨,$t_\text{step}$ 随 $c$ 线性上升 → 在飞 decode 的 ITL 被顶到 $t_\text{step}$。

**stall 量化**:在飞请求本应每 $t_\text{decode}$ 出一个 token,被 prefill 占用 $t_\text{prefill}$ 后,这段时间的 ITL 尖刺:

$$
\text{ITL}_\text{spike} \approx t_\text{prefill} \gg t_\text{decode},\qquad \text{放大倍数} \approx \frac{t_\text{prefill}}{t_\text{decode}} \propto \frac{S_\text{prompt}}{1}
$$

治理 = 把 $c$ 钳住或移走:
- **chunk**:强制 $c \le C_\text{chunk}$,使 $t_\text{step}$ 有上界(见 [[042 chunked prefill：切块融合|chunked prefill]]),这是把干扰**摊平**。
- **分离**:让 prefill 与 decode 各自独占实例,$c$ 与 $d$ 不再同步,这是把干扰**消除**(代价:跨实例传 KV)。

## 图

![[pd-干扰定量.png]]

![[sched-prefill干扰stall.png]]

## 代码

复现 stall,并对比三种调度策略:

```python
def step_latency(prefill_tokens, decode_count, alpha=0.05, base=2.0):
    # 单步耗时 ≈ alpha * 总token + 固定开销（ms）；prefill_tokens 主导
    return alpha * (prefill_tokens + decode_count) + base

# ❌ prefill 优先：长 prompt 一来就独占整步 → 在飞 decode stall
def naive_prefill_first(long_prompt_len, inflight):
    return step_latency(long_prompt_len, inflight)        # 例：0.05*4096 ≈ 207ms stall

# ✅ chunked：每步只放一块，单步耗时被预算钳住
def chunked(long_prompt_len, inflight, chunk=512):
    n_steps = (long_prompt_len + chunk - 1) // chunk
    return [step_latency(chunk, inflight) for _ in range(n_steps)]  # 每步 ≈ 30ms，无尖刺

# ✅ PD 分离：decode 实例完全不混 prefill
def disaggregated(inflight):
    return step_latency(0, inflight)                      # decode 步永远 ~base+，零干扰

if __name__ == "__main__":
    print("naive  :", naive_prefill_first(4096, 60), "ms  ← stall")
    print("chunked:", chunked(4096, 60)[0], "ms/step  ← 平稳")
    print("disagg :", disaggregated(60), "ms/step  ← 零干扰")
```

`❌` prefill 优先让单个长 prompt 的 ITL 尖刺直接传导给所有在飞请求;`✅` chunked 把尖刺摊平成多个小步、`✅` 分离把 prefill 彻底移出 decode 实例。注意 chunked 在单实例内零额外硬件成本,分离则需要额外 GPU + KV 传输带宽。


![[sched-043stall传导.png]]

## 面试高频

- **Q:什么是 generation stall?** 长 prefill 与在飞 decode 混在同一步时,prefill 大 GEMM 拖长该步,在飞 decode 这段时间出不了 token,表现为 ITL/TPOT 周期性尖刺。
- **Q:根因是什么?** prefill 计算受限(又宽又大的 GEMM)、decode 访存受限(又细又长),二者抢同一步算力;而连续批"新请求即补位"必然引入 prefill,故干扰天生。
- **Q:为什么 prefill 优先调度会害死延迟?** prefill 时长随 prompt 长度任意增长,prefill 优先意味着任意时刻可能插入一个超长 prefill,在飞请求的 TPOT 上界因此不可控(Sarathi 论文核心论点)。
- **Q:chunk 和分离怎么选?** 单实例、流量混杂、不想加硬件 → chunked prefill(摊平干扰);超大规模、prefill/decode 负载差异大、要独立扩容 → PD 分离(消除干扰但需传 KV)。

## 关键事实

- **generation stall** 由 prefill/decode 混批引起,是 prefill-prioritizing 调度器的固有缺陷;Sarathi-Serve(Agrawal et al., OSDI 2024)以此为出发点提出 stall-free 调度。
- stall 尖刺幅度 ≈ prefill 时长,随 prompt 长度线性增长;长上下文(几十 K token)场景下尤其严重。
- 两条主流治理路线截至 2025 共存:**chunked prefill**(单实例摊平,vLLM/SGLang 默认)与 **PD 分离**(DistServe、Splitwise、MoonCake 等多实例消除);二者也可叠加使用。
