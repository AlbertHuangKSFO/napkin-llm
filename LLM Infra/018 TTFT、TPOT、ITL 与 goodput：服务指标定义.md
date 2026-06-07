[[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|LLM 服务指标]]是衡量推理服务质量的一套精确定义:**TTFT**(首 token 延迟)= 请求到达到吐出第 1 个 token 的时间,由 [[013 Prefill 阶段：计算受限|Prefill]] 决定;**TPOT**(每输出 token 时间)= decode 阶段相邻 token 的平均间隔,等价于 **ITL**(inter-token latency),由 [[014 Decode 阶段：访存受限|Decode]] 决定;**端到端延迟 E2E** ≈ TTFT + (输出 token 数−1)×TPOT;**throughput** = 单位时间产出 token/请求数;**goodput** = 单位时间内**同时满足全部 SLO**(如 TTFT&lt;200ms 且 TPOT&lt;50ms)的请求数。goodput 才是服务优化的真正目标,它把 [[017 吞吐与延迟：根本性张力|吞吐延迟张力]]收敛成一个可优化的数。

## 直觉

四个指标对应用户体验的不同侧面:

- **TTFT** = "等了多久才开始看到字"——聊天首字反应速度,慢了像卡死。
- **TPOT/ITL** = "字往外蹦的流畅度"——太慢像逐字打字机,太快人眼也读不过来。
- **E2E** = "整条回答花了多久"——长回答里 TPOT 是主导项。
- **throughput** = 运营视角的"机器总产能";**goodput** = "其中真正让用户满意(达标)的那部分产能"。

关键反直觉:**goodput ≤ throughput**。一台机器可以吞吐很高,但若一半请求 TTFT 或 TPOT 超标,这一半就是"废产"——付了算力却没换来达标体验。盲目堆 batch 抬 throughput,常常把 goodput 抬反。

## 例子

聊天服务 SLO:**TTFT &lt; 200 ms 且 TPOT &lt; 50 ms**,要求 P90 达标。

- 某配置实测:throughput = **10 req/s**,但其中 TTFT 超标 2 req/s、TPOT 超标 1.5 req/s(部分重叠),最终**两条都满足**的只剩 **6.5 req/s** → **goodput = 6.5 req/s**,3.5 req/s 是废产。
- 端到端例子:prompt 2k token,prefill 50 ms →TTFT = 50 ms;生成 200 token,TPOT 40 ms →E2E ≈ $50 + 199\times 40 = 8010$ ms ≈ 8 秒。
- 流式体验:TPOT 40 ms ≈ 25 token/s ≈ 人类舒适阅读速度;若 TPOT 升到 100 ms(10 tok/s),用户明显感到卡顿,即便 throughput 更高。

## 原理

精确定义(设请求 $i$ 输出 $n_i$ 个 token):

$$
\text{TTFT}_i = t^{(1)}_i - t^\text{arrive}_i,\qquad
\text{TPOT}_i = \text{ITL}_i = \frac{t^{(n_i)}_i - t^{(1)}_i}{n_i - 1}
$$

$$
\text{E2E}_i = \text{TTFT}_i + (n_i-1)\,\text{TPOT}_i,\qquad
\text{Throughput} = \frac{\sum_i n_i}{T}\ \text{(tok/s)}\ \text{或}\ \frac{\#\text{req}}{T}\ \text{(req/s)}
$$

**SLO attainment** = 满足约束的请求占比;**goodput** = SLO 内的达标吞吐:

$$
\text{Goodput} = \max\Big\{\text{req/s}\ :\ \mathbb{P}\big[\text{TTFT}\le \tau_\text{ttft}\ \wedge\ \text{TPOT}\le \tau_\text{tpot}\big]\ge \alpha\Big\}
$$

($\alpha$ 常取 P90/P99)。注意 ITL 严格说是逐对相邻 token 间隔的序列,TPOT 是其平均;多数文献(Anyscale/DistServe,2024–2025)将 TPOT 与 ITL 视为等价。goodput 的难点在于它同时约束**由不同阶段决定**的 TTFT(prefill)与 TPOT(decode),二者对 batch 反应相反 → 单一系统难两头讨好,这是 PD 分离的动机。

## 图

![[pd-服务指标时间轴.png]]

![[pd-goodput概念图.png]]

![[pd-018吞吐与goodput废产.png]]

## 代码

从每请求的 token 时间戳算全套指标 + goodput,附 `❌vs✅`:

```python
import numpy as np

def metrics(arrive, token_times):
    # token_times: 该请求各 token 产出的时间戳(升序)
    ttft = token_times[0] - arrive
    tpot = (token_times[-1] - token_times[0]) / (len(token_times) - 1) if len(token_times) > 1 else 0.0
    e2e  = token_times[-1] - arrive
    return ttft, tpot, e2e            # 单位秒

def goodput(reqs, ttft_slo, tpot_slo, window_s):
    # reqs: [(arrive, token_times), ...]
    ok = 0
    for arrive, tt in reqs:
        ttft, tpot, _ = metrics(arrive, tt)
        if ttft <= ttft_slo and tpot <= tpot_slo:   # ✅ 两条 SLO 都满足才算达标
            ok += 1
    return ok / window_s              # 达标请求 / 秒

# ❌ 错误：拿系统总吞吐当 KPI（throughput = len(reqs)/window），把超标请求也算进去
# ✅ 正确：goodput 只数同时满足 TTFT 与 TPOT 的请求 —— 反映真实可用产能
gp = goodput(reqs, ttft_slo=0.2, tpot_slo=0.05, window_s=60)  # P90 另需分位过滤
```

`❌` 用裸 throughput 当成绩单,会把违反 SLO 的废产也计入、掩盖体验崩坏;`✅` goodput 只计同时满足 TTFT 和 TPOT 的请求,且通常按 P90/P99 分位卡。

## 面试高频

- **Q:TTFT、TPOT、ITL、E2E 分别由哪个阶段决定、怎么算?** TTFT←prefill(到第 1 token);TPOT=ITL←decode(相邻 token 平均间隔);E2E = TTFT + (n−1)·TPOT。
- **Q:throughput 和 goodput 区别?** throughput 是总产出;goodput 只数满足全部 SLO 的达标产出,**goodput ≤ throughput**,是真正该优化的目标。
- **Q:为什么盲目堆 batch 会降 goodput?** batch 大→throughput 升但 TPOT/TTFT 升,越过 SLO 后达标请求变少,goodput 反降(DistServe 2024 的核心论点)。
- **Q:TPOT 和 ITL 有区别吗?** 严格说 ITL 是逐对相邻间隔序列、TPOT 是其平均;实践中(Anyscale 等)常视为等价。
- **Q:goodput 为什么难优化?** 它同时约束 prefill 决定的 TTFT 和 decode 决定的 TPOT,二者对 batch 反应相反,单系统两头难顾 → PD 分离把两阶段放不同 GPU 各自调度。

## 关键事实

- **goodput = 满足 SLO 的达标吞吐**,概念由 **DistServe(Hao AI Lab,2024)** 在 LLM 服务语境正式提出,主张 prefill/decode 分离以最大化它。
- 典型 SLO 形如 "P90 TTFT &lt; 200 ms 且 P90 TPOT &lt; 50 ms";goodput 即此约束下的最大 req/s(Anyscale/BentoML 文档,2024–2025)。
- **TPOT ≡ ITL** 在多数 2025 文献中等价;TTFT 由 prefill(计算受限)定,TPOT 由 decode(访存受限)定,这是把指标接回[[012 自回归推理全流程：一个 token 的旅程|两阶段本质]]的关键桥。
