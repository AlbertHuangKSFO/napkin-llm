[[049 DistServe：goodput 优先的解耦|DistServe]](Zhong 等,OSDI 2024,arXiv 2401.09670)是第一个系统性论证 [[048 为何分离 prefill 与 decode|PD 分离]] 并以 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|goodput]] 为优化目标的工作。它把 [[013 Prefill 阶段：计算受限|Prefill]] 与 [[014 Decode 阶段：访存受限|Decode]] 分到不同 GPU 池,**各自独立选并行策略**,消除两阶段干扰;再根据集群带宽放置两阶段、最小化 [[053 KV 传输：NIXL、点对点与带宽|KV 传输]] 成本。论文标题即口号:Throughput is not all you need——该最大化的是满足 SLO 的 goodput,而非裸吞吐。

## 直觉

DistServe 的洞察像工厂分线:一条生产线既做"重切削"(prefill)又做"精装配"(decode),换型频繁、互相等待;不如**拆成两条专线**,各自上最合适的机器、各自定节拍。更关键的是它换了 KPI:不看"机器一共出了多少件",而看"**多少件按时且合格交付**"(goodput)——盲目堆 batch 把吞吐拉高,却让一半请求 TTFT/TPOT 超标,那一半是废产。

## 例子

论文给的量级(各种模型/应用/SLO 下):

- 相比 SOTA 聚合系统,DistServe 在满足 SLO(>90% 请求达标)前提下,可服务 **最多 7.4× 更多请求**,或在同负载下扛 **12.6× 更严的 SLO**。
- 并行策略分化的例子:prefill 池为压低 TTFT 选较高张量并行(如 TP=4),decode 池为攒吞吐选较大 batch + 较低并行(如 TP=2);这在聚合系统里**无法分别设置**。
- 放置:若 prefill 与 decode 在同节点 NVLink 域内,KV 传输近乎免费;跨节点则 DistServe 按带宽决定是否分离,避免传输抵消收益。

## 原理

DistServe 把服务建模为:在 TTFT、TPOT 约束下,**分别**为两阶段搜索 (并行配置, 资源量) 以最大化 goodput:

$$
\max_{\;\theta_p,\theta_d,\,n_p,n_d}\ \text{Goodput}
\quad \text{s.t.}\quad
\text{TTFT}(\theta_p)\le \tau_{\text{ttft}},\ \ \text{TPOT}(\theta_d)\le \tau_{\text{tpot}}
$$

其中 $\theta_p,\theta_d$ 是两阶段各自的并行方案,$n_p,n_d$ 是实例数(对应 [[054 PD 配比与独立扩缩|PD 配比]])。聚合系统被迫 $\theta_p=\theta_d$ 且共享 batch,可行域被严重压缩;解耦把它拆成两个**几乎独立**的子问题。放置项再加一个传输惩罚:

$$
\text{cost}_{\text{place}} \approx \frac{S_{\text{kv}}}{B_{\text{link}}}\quad(\text{同 NVLink 域} \to B_{\text{link}}\ \text{大,惩罚小})
$$

## 图

![[disagg-DistServe架构.svg]]

![[disagg-049goodput非吞吐.svg]]

![[disagg-049独立并行搜索.svg]]

## 代码

DistServe 开源实现(LLMServe/DistServe)按角色起 worker;下面示意其"两阶段各自并行 + KV 接力"的配置思路:

```python
# ❌ 聚合:prefill 与 decode 共用同一并行方案与同一批,SLO 紧时顾此失彼
# engine = Engine(model, tp=4, enable_chunked_prefill=True)

# ✅ DistServe:两池各自的并行度,goodput 为目标搜索配比
prefill_cfg = dict(role="prefill", tp=4, pp=1,   # 压 TTFT:高并行、小批
                   target="minimize_ttft")
decode_cfg  = dict(role="decode",  tp=2, pp=1,   # 攒吞吐:大批、低并行
                   max_batch=256, target="maximize_throughput")

# 放置:同 NVLink 域则 KV 传输近免费;否则按带宽决定是否值得分离
placement = plan_placement(prefill_cfg, decode_cfg,
                           bandwidth_aware=True)  # ✅ 最小化 KV 传输成本
```

`❌` 共用并行方案,prefill 想要的高并行会拖累 decode 吞吐;`✅` 两阶段各搜各的最优配置,再按带宽放置以压低 KV 传输,目标始终是 goodput 而非裸 throughput。

## 面试高频

- **Q:DistServe 的一句话贡献?** 首个以 goodput(满足 TTFT+TPOT 的达标吞吐)为目标、系统化做 prefill/decode 分离的工作,论文核心论点"throughput is not all you need"。
- **Q:它比聚合系统强多少?** 论文报告最高 **7.4× 更多请求**或 **12.6× 更严 SLO**(>90% 请求达标),OSDI 2024。
- **Q:分离后并行策略怎么变?** 两阶段**独立搜索**并行配置:prefill 偏高并行压 TTFT,decode 偏大 batch 攒吞吐,聚合系统做不到这种分化。
- **Q:DistServe 怎么处理 KV 传输?** 带宽感知放置:尽量把配对的 prefill/decode 放进同一高带宽域(NVLink),跨节点时纳入传输惩罚再决定是否分离。

## 关键事实

- **DistServe**,Zhong 等,**OSDI 2024**,arXiv **2401.09670**;Hao AI Lab(UCSD)。
- 核心数字:满足 SLO 下 **最高 7.4× 请求量**或 **12.6× 更严 SLO**;开源仓库 LLMServe/DistServe。
- 三板斧:① 两阶段分到不同 GPU 消除干扰;② 各自 co-optimize 资源与并行;③ 带宽感知放置最小化 [[053 KV 传输：NIXL、点对点与带宽|KV 传输]]。它把 goodput 从概念推成了可优化目标,后续 [[051 Mooncake：KVCache 中心解耦架构|Mooncake]]、[[052 NVIDIA Dynamo：分布式推理框架与 SLO Planner|Dynamo]] 沿用此范式。
