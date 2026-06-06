这是 [[LLM Infra]] 域回答「要买/开几张卡」的反推题,是 [[060 数据并行与副本扩展|副本扩展]]和 [[101 自动扩缩：HPA、KEDA、scale-to-zero 与冷启动|自动扩缩]]的前置算账。**容量规划(capacity planning)**=从业务给的**目标 QPS × 平均 token 数 × SLO**,反推出**每卡可用吞吐**,再除出 **GPU 台数**,最后乘上峰值冗余与排队缓冲。一句话:把 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|服务指标]]里的延迟约束,翻译成「ρ 不能跑到 1」的利用率上限,从而决定要堆多少卡。它和 [[110 成本与 TCO：每百万 token 成本怎么算|每百万 token 成本]]是一体两面:卡数定了,成本就定了。

类比:开餐厅排班。你不会按「全天平均客流」配厨师——午高峰会瞬间排长队赶走客人(尾延迟爆)。你按**高峰客流**配,还要留**机动人手**应对突发团客(流量 burst)和有人请假(故障冗余)。而且厨房不能 100% 满负荷转——一满就出餐巨慢(排队论:ρ→1 等待时间发散)。所以**永远按峰值 + 余量配,不按平均**。

小数字反推:目标峰值 100 QPS,平均每请求输出 500 token → token 需求 $100\times500=50{,}000$ tok/s。单卡峰值吞吐 2000 tok/s,但要满足 SLO 不能跑满,取**可用利用率 0.6** → 每卡可用 $2000\times0.6=1200$ tok/s。基础卡数 $\lceil 50000/1200\rceil=42$ 卡。再乘**突发系数** 1.5 和**故障冗余** 1.15 → 实际 $42\times1.5\times1.15\approx 73$ 卡。这就是为何「平均要 16 卡」的业务,遇到陡峭突发可能真要上百卡——突发越凶,冗余倍数越大。

$$
N_{\text{gpu}}=\Big\lceil\frac{\text{QPS}_{\text{peak}}\times \bar{T}_{\text{out}}}{\text{Thr}_{\text{gpu}}\times u_{\text{SLO}}}\Big\rceil\times f_{\text{burst}}\times f_{\text{fail}},\qquad
\rho=\frac{\lambda S}{c}<1
$$

为什么有 $u_{\text{SLO}}$ 这个「不能跑满」的扣减?排队论(M/G/c)告诉你,每卡的负荷 $\rho=\lambda S/c$ 越接近 1,等待时间越是**陡峭发散**——利用率从 0.7 提到 0.9,P99 等待可能翻几倍。所以 SLO 工作点通常压在 $\rho\approx0.6\!-\!0.8$,把延迟尾巴留在悬崖之前。$\text{Thr}_{\text{gpu}}$ 怎么来?靠 [[107 基准：MLPerf Inference 与 InferenceMAX|基准]]实测,且**强依赖** [[LLM/078 推理算力、吞吐与延迟、Roofline|Roofline]]、batch、量化与 prefill/decode 配比;输入/输出 token 比例不同,每卡吞吐差很多。规划完不是静态买死:配 [[101 自动扩缩：HPA、KEDA、scale-to-zero 与冷启动|自动扩缩]]按日内曲线伸缩,基线扛峰值、弹性吃尖峰。

![[obs-容量规划计算.svg]]

![[obs-113利用率悬崖.svg]]

```python
import math
def plan_gpus(qps_peak, avg_out_tok, thr_gpu, u_slo=0.6,
              f_burst=1.5, f_fail=1.15):
    demand = qps_peak * avg_out_tok          # tok/s 总需求
    per_gpu = thr_gpu * u_slo                 # 每卡可用吞吐(留 SLO 余量)
    base = math.ceil(demand / per_gpu)        # 基础卡数
    return base, math.ceil(base * f_burst * f_fail)

base, total = plan_gpus(qps_peak=100, avg_out_tok=500, thr_gpu=2000)
print(base, total)   # 42 73  —— 基础 42 卡,含突发+故障冗余约 73 卡
```

```text
❌ 按"平均 QPS / 单卡峰值吞吐"配卡,且把利用率打满到 1.0
   → 高峰一来排队论发散:ρ→1 等待时间陡增,P99 TTFT/TPOT 爆穿 SLO
✅ 按峰值 QPS、每卡用 thr×u_SLO(留 ρ≤0.6–0.8 余量),再乘突发×故障冗余
   → 留住延迟尾巴;配 HPA/KEDA 跟随日内曲线弹性扩缩,基线扛峰值
```

## 面试高频
- **怎么从 QPS 反推 GPU 数?** $N=\lceil(\text{QPS}_\text{peak}\times\bar T_\text{out})/(\text{Thr}_\text{gpu}\times u_\text{SLO})\rceil\times f_\text{burst}\times f_\text{fail}$;先算 token 需求,除以每卡可用吞吐,再乘冗余。
- **为什么不能按平均 QPS 配?** 高峰瞬时负载远超平均,排队论下 ρ→1 等待时间发散;必须按峰值 + 余量。
- **为什么每卡吞吐要打折扣($u_\text{SLO}$)?** 利用率越接近 1,P99 延迟越陡升(M/G/c)。SLO 工作点压在 ρ≈0.6–0.8,留延迟余量。
- **峰值冗余和故障冗余分别是什么?** 突发系数 $f_\text{burst}$(1.3–2×)应对流量尖峰;故障冗余 $f_\text{fail}$(N+1/+15%)应对卡/节点失效。突发越凶,冗余倍数越大。
- **每卡吞吐怎么定?** 靠基准实测,依赖 batch、量化、prefill/decode 配比与输入/输出 token 比例;不是常数,要按真实流量画像测。

## 关键事实
- 反推链:目标 QPS × 平均输出 token = tok/s 需求;÷(每卡吞吐 × SLO 利用率)= 基础卡数;× 突发 × 故障冗余 = 部署卡数。
- 必须按峰值而非平均配;每卡负荷 $\rho=\lambda S/c$ 须 <1,小幅提利用率导致等待时间巨增(**2025–2026**,inference-fleet-sim)。
- 突发流量代价巨大:同等平均量下,bursty 工作负载(尖峰 400 req/min)可能需 ~125 GPU,而平滑流量仅 ~16 GPU(**2025**)。
- 显存侧留 ~10–20% headroom 给 KV cache、batching 与引擎开销。
- 现代做法:M/G/c 排队 + 离散事件仿真求「满足 P99 TTFT SLO 的最小成本配置」(**2026**,inference-fleet-sim);配 HPA/KEDA 弹性扩缩。
