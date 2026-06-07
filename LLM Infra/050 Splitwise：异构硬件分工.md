[[050 Splitwise：异构硬件分工|Splitwise]](Patel 等,Microsoft,ISCA 2024,arXiv 2311.18677)把 [[048 为何分离 prefill 与 decode|PD 分离]] 推到**异构硬件**这一步:既然 [[013 Prefill 阶段：计算受限|Prefill]] 吃算力、[[014 Decode 阶段：访存受限|Decode]] 吃带宽,那就让 prefill 跑在**强算力的新卡**(如 H100)、decode 跑在**够带宽但更便宜/省电的旧卡**(如 A100),按阶段把硬件用在刀刃上。核心收益是 cost 与 power 维度的优化,而非单纯延迟——这是 [[018 TTFT、TPOT、ITL 与 goodput：服务指标定义|goodput]] 之外,PD 分离带来的"硬件错配套利"。

## 直觉

类比搬家公司:**打包(prefill)**需要壮劳力短时爆发(高算力);**送货(decode)**只要一辆够大的厢式货车慢慢跑(够带宽即可,不需顶配)。让顶配卡车去打包是浪费。Splitwise 的洞察:decode 不需要最新 GPU 的峰值算力,把它丢给上一代便宜卡,既不掉性能,又省钱省电;把贵卡的算力全留给真正吃算力的 prefill。

## 例子

论文报告的量级(ISCA 2024):

- 同 **成本与功耗预算**下,Splitwise 异构集群可达 **2.35× 更高吞吐**。
- 或换个目标:**1.4× 更高吞吐**的同时 **降低 20% 成本**。
- 硬件错配的直观账:decode 逐 token 主要在读权重 + [[LLM/102 KV-Cache|KV-Cache]],瓶颈是 HBM 带宽。A100(~2 TB/s)对 H100(~3.35 TB/s)带宽差距,远小于二者算力差距与价格/功耗差距 → 让 A100 接 decode,性价比高;H100 的昂贵算力专供 prefill 才不浪费。

## 原理

设单位算力成本 prefill 卡为 $c_p$、decode 卡为 $c_d$($c_d < c_p$),两阶段各自所需机器数 $n_p, n_d$。聚合系统被迫两阶段同卡(都用贵卡),异构分离的成本是:

$$
\text{Cost}_{\text{hetero}} = n_p\,c_p + n_d\,c_d
\;<\;
\text{Cost}_{\text{homo}} = (n_p+n_d)\,c_p
$$

只要 decode 在便宜卡上仍满足 TPOT,即 $\text{TPOT}_{\text{decode-card}} \le \tau_{\text{tpot}}$ 成立,这笔替换就净赚。功耗同理——decode 卡可在更低功耗档运行,因为它不需打满算力:

$$
\text{Goodput per Watt} \uparrow\quad\text{当 decode 迁到低功耗、够带宽的卡}
$$

## 图

![[disagg-Splitwise异构分工.png]]

![[disagg-050硬件错配套利.png]]

![[disagg-050异构集群部署.png]]

## 代码

Splitwise 的部署本质是给两池指定不同卡型,再用 KV 传输接力:

```yaml
# ❌ 同构:prefill 与 decode 都堆在 H100 上,decode 浪费昂贵算力
# pool: { type: unified, gpu: H100, count: 8 }

# ✅ Splitwise 异构:贵卡做 prefill,便宜卡做 decode
prefill_pool:                 # 吃算力 → 上最强卡
  gpu: H100
  count: 4
  role: kv_producer
decode_pool:                  # 吃带宽 → 上够用的便宜/省电卡
  gpu: A100                   # 带宽够、单价低、功耗低
  count: 8
  role: kv_consumer
  power_cap: low              # decode 不需峰值算力,可压功耗
kv_transfer: { backend: rdma } # prefill→decode 传 KV(见 NIXL 笔记)
```

`❌` 用同一种贵卡跑两阶段,decode 把昂贵算力闲置成带宽搬运工;`✅` 按阶段配卡型,贵算力归 prefill、便宜带宽归 decode,在同成本/功耗下抬吞吐。

## 面试高频

- **Q:Splitwise 和 DistServe 的区别?** 都做 PD 分离,但 Splitwise 强调**异构硬件分工**(prefill 用强算力卡、decode 用便宜带宽卡)以优化 cost/power;DistServe 更侧重 goodput 与并行策略搜索。
- **Q:为什么 decode 可以用便宜的旧卡?** decode 访存受限,瓶颈是 HBM 带宽而非算力;旧卡带宽够、价格与功耗更低,放 decode 性价比高,而把贵卡的峰值算力留给吃算力的 prefill。
- **Q:Splitwise 的代表数字?** ISCA 2024:同成本同功耗下 **2.35× 吞吐**;或 **1.4× 吞吐 + 降 20% 成本**。
- **Q:异构分离有什么新代价?** 除了 PD 分离本身的 KV 传输,异构还增加调度/运维复杂度(两种卡型的扩缩、配比、故障域),且 KV 仍要跨机传输。

## 关键事实

- **Splitwise**,Patel 等,**Microsoft / UW**,**ISCA 2024**,arXiv **2311.18677**。
- 核心数字:同成本同功耗 **2.35× 吞吐**,或 **1.4× 吞吐 & −20% 成本**(论文报告)。
- 关键论点:token generation(decode)**不需要最新 GPU 的算力**,可用低功耗低成本卡;把异构性当作 PD 分离的一等收益。它与 [[050 Splitwise：异构硬件分工|本篇]] 同源的思路也出现在 Mooncake、[[062 Wide-EP：DeepSeek、Kimi 在 H100、H200 上的部署|Wide-EP]] 的硬件分层部署中。
