[[105 SLO、SLA 设计：为推理定指标|SLO、SLA 设计]]是把 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|服务指标]]从「实验室数字」变成「生产合约」的那一步。**SLI(Service Level Indicator)** 是实测出来的指标(TTFT、TPOT、错误率);**SLO(Objective)** 是你给自己定的内部目标(「TTFT p95 ≤ 300ms」);**SLA(Agreement)** 是对外的合约,违反要赔钱,因此口径比 SLO 更宽松、留缓冲。关键反直觉:**SLO 必须用百分位(p95/p99)而非平均值**,因为平均值会被尾部隐藏——avg 200ms 的系统可以 p99 高达 3000ms,1% 用户体验糟糕得多。SLO 之外配 **错误预算(error budget)**:99.9% 的目标意味着 0.1% 的请求允许破,这个预算决定你能不能激进发版。它和 [[047 准入控制与排队论：队列长度到延迟|排队论]]、[[108 监控：GPU 利用率、KV 占用、排队、p99|监控指标]]、[[110 成本与 TCO：每百万 token 成本怎么算|成本 TCO]]共同构成生产服务的「合约—度量—兜底」闭环。

## 直觉

把 SLO 想成**餐厅对上菜时间的承诺**。

- 「平均 10 分钟上菜」听着不错,但如果 1% 的桌子等了 1 小时,这些客人永远不会再来——**平均值骗人,尾部杀人**。所以承诺要写成「95% 的桌子 15 分钟内上菜」(p95),这才约束了体验下限。
- **SLA 是写进菜单的赔付条款**(「超 20 分钟免单」),比厨房内部的 SLO 目标松——内部死守 15 分钟,对外只敢承诺 20 分钟,留 5 分钟缓冲应对意外。
- **错误预算**:你承诺 99.9% 准时,等于「这个月允许 0.1% 的桌子超时」。预算还有余 → 敢上新菜试验;预算烧光 → 厨房冻结一切变更,先稳住。

推理服务一模一样:TTFT 是「首道菜多快上」,TPOT/ITL 是「后续菜上得顺不顺」,都得用百分位立约。

## 例子

某聊天服务,统计一天的 TTFT(ms):

| 分位 | TTFT | 解读 |
|---|---|---|
| p50(中位) | 120 | 一半用户体验很好 |
| avg(平均) | 200 | 被尾部拉高,**别拿它立 SLO** |
| p95 | 480 | SLO 立在这:95% 用户 ≤ 480ms |
| p99 | 1500 | 1% 用户等 1.5s——尾部真相 |
| p99.9 | 4200 | 0.1% 用户等 4s+,常是排队/抢占 |

- **百分位 vs 平均的鸿沟**:avg 200ms 看着达标,但 p99=1500ms 意味着每天 100 万请求里有 1 万次糟糕体验。只盯平均会完全错过。
- **SLO 设计**:交互式定「TTFT p95 ≤ 500ms 且 ITL p95 ≤ 50ms/token,错误率 ≤ 1%,窗口 28 天」。对外 **SLA 放宽到 p99 ≤ 800ms**,留缓冲。
- **错误预算账**:SLO=99.9% 可用 → 预算 = 0.1%。日均 100 万请求 × 28 天 = 2800 万,允许破 **28000 个**。某次发版一天就烧掉 15000 → 预算燃烧过快 → 触发回滚 + 冻结。

## 原理

**百分位(percentile)** 定义:$p$ 分位 $x_p$ 满足

$$
\Pr[X \le x_p] = \frac{p}{100}
$$

即 $p\%$ 的样本 $\le x_p$。生产里由直方图(histogram)的桶估算,而非排序全量(成本太高)。**为什么用 p99 而非 avg**:延迟分布**右偏重尾**,$\text{avg} \ll p99$;SLO 要保的是「绝大多数用户」的体验下限,只有高百分位能约束尾部。

**错误预算(error budget)**:窗口内允许的失败比例

$$
\text{Budget} = (1 - \text{SLO}) \times N_{\text{total}}
$$

**燃烧率(burn rate)** = 实际消耗速度 / 均匀消耗速度。burn rate = 1 表示刚好月底用完;burn rate = 14.4 表示 2 小时烧光一个月的预算(常用多窗口告警阈)。

$$
\text{burn rate} = \frac{\text{实际错误率}}{1 - \text{SLO}}
$$

燃烧率→预算耗尽时间→告警窗口/等级的对照(含 14.4 为何取这个数,及双窗口告警):

![[orch-105燃烧率耗尽窗口.png]]

**goodput 视角**:SLO 不只是「不出错」,还要「够快」。把同时满足 TTFT 与 TPOT 阈值的请求才算达标,达标吞吐就是 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|goodput]];定 SLO 等于定义「什么样的请求算数」。

## 图

![[obs-SLO设计分层.png]]

![[obs-105尾延迟与燃烧率.png]]

## 代码

从 Prometheus 直方图算分位并核对 SLO:

```python
# ❌ 反模式:拿平均值定 SLO / 报达标
def slo_ok_avg(latencies_ms):
    avg = sum(latencies_ms) / len(latencies_ms)
    return avg < 500          # avg=200 通过,却掩盖 p99=1500ms 的糟糕尾部

# ✅ 用百分位定 SLO,分别约束 TTFT 与 ITL
import numpy as np
def slo_report(ttft_ms, itl_ms, err_count, total, window_slo=0.999):
    p95_ttft = np.percentile(ttft_ms, 95)
    p99_ttft = np.percentile(ttft_ms, 99)
    p95_itl  = np.percentile(itl_ms, 95)
    err_rate = err_count / total
    budget   = (1 - window_slo) * total          # 错误预算(请求数)
    burn     = err_rate / (1 - window_slo)       # 燃烧率
    return {
        "TTFT_p95": p95_ttft, "TTFT_p99": p99_ttft, "ITL_p95": p95_itl,
        "slo_pass": p95_ttft <= 500 and p95_itl <= 50 and err_rate <= 0.01,
        "budget_left": budget - err_count,        # 预算余量
        "burn_rate": burn,                        # >1 烧太快;>14 紧急
    }
```

```promql
# Grafana:TTFT p99(对 histogram 用 histogram_quantile)
histogram_quantile(0.99,
  sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))
```

`❌` 用平均值会让破 SLO 的服务「看起来达标」,尾部用户被牺牲而不自知;`✅` 用百分位 + 分维度(TTFT 与 ITL 各立约) + 错误预算,才能既守住体验下限,又量化「还能冒多大险发版」。

## 面试高频

- **Q:SLI / SLO / SLA 区别?** SLI 是实测指标(TTFT、错误率);SLO 是内部目标(p95 ≤ 300ms);SLA 是对外合约、违约赔付、口径比 SLO 宽松留缓冲。三者是「度量→目标→承诺」。
- **Q:为什么 SLO 用 p95/p99 不用平均?** 延迟分布右偏重尾,avg 远小于 p99;SLO 要保绝大多数用户的体验下限,只有高百分位约束尾部。avg 200ms 可同时 p99=3000ms。
- **Q:错误预算是什么,有什么用?** (1−SLO)×总请求数 = 允许失败的额度;预算有余可激进发版,燃烧过快(burn rate 高)则冻结变更/回滚。把可靠性和迭代速度量化挂钩。
- **Q:推理服务该定哪些 SLO?** TTFT(首 token,交互延迟)、TPOT/ITL(流式顺滑度)、可用性/错误率、goodput(达标吞吐)。分别立百分位阈值。
- **Q:交互式和批处理的 SLO 怎么分?** 交互式严控 TTFT p95 + ITL,低利用率换低延迟;批处理无 TTFT 约束,只看总吞吐/成本,ρ 可拉到 0.9+。同一引擎按工作负载分级。
- **Q:SLA 为什么比 SLO 宽松?** SLA 违约要赔钱,需留缓冲吸收抖动/故障;内部 SLO 死守更严的目标,使得即便偶发劣化也不至于触碰对外合约。

## 关键事实

- **SLO 一律用百分位(p95/p99/p99.9)**,不用平均值——延迟右偏重尾,平均值掩盖尾部;生产用直方图桶(histogram)估分位,`histogram_quantile` 是 PromQL 标配。
- **错误预算 = (1−SLO)×总量**,配燃烧率(burn rate)多窗口告警(常见阈 burn=14.4 表示 ~2h 烧光月预算);Google SRE 体系把它作为「可靠性 vs 迭代速度」的硬约束(2025 仍是业界标准实践)。
- **推理 SLO 至少四项**:TTFT、TPOT/ITL、可用性/错误率、goodput;交互式严控 TTFT+ITL,批处理只看吞吐/成本,按工作负载分级定不同阈值(2025–2026 主流引擎 vLLM/TGI 均原生暴露这些指标)。
