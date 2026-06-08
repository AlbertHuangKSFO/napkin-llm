这是 [[LLM Infra]] 域讲**跨节点网络**(scale-out)的选型,承接 [[009 节点与机架：PCIe、GB200 NVL72、scale-up 与 scale-out|scale-up/out]] 的 scale-out 一侧,是 [[066 NCCL：集合通信库与原语|NCCL]] 跨节点传输层的物理底座,参见 [[LLM/069 数据并行与 AllReduce|AllReduce]]。一句话:**InfiniBand 和 RoCE 都跑 RDMA(网卡直读对端显存、绕开 CPU),区别在「谁保证不丢包」——IB 用信用流控天生无损、延迟低、开箱即用但贵且单一供应商;RoCE 把 RDMA 封进标准以太网,便宜、复用以太生态,但必须人工配 PFC+ECN 才无损,配错就暂停风暴慢 5–10×。**

类比:两条都是"VIP 直达通道"(RDMA,绕开 CPU 这个前台)。**IB** 像专用磁悬浮:轨道、车、站台全是一家造的,信用流控=进站前先确认有空位才放行,天生不堵不丢,但建一条独立线路很贵。**RoCE** 像在现有高速公路(以太网)上划 VIP 车道:省钱、用现成路网,但高速本来会堵会"丢车",得靠 PFC(红灯硬拦住后车)+ ECN(提前亮黄灯让你减速)凑出"不丢";红绿灯配错,一处堵全网瘫(暂停风暴)。

RDMA 是两者共同的底:网卡(HCA/NIC)直接读写对端 GPU 显存(配 GPUDirect RDMA),CPU 不参与拷贝,这是低延迟高带宽的根本。差异全在**无损怎么实现**:IB 信用流控(发送前先要额度,没额度不发→不会溢出丢包);RoCE v2 把 IB 语义封进 UDP/IP 跑以太网,以太网本会丢包,所以要 **PFC**(按优先级暂停,硬防丢但会头阻塞)+ **ECN**(拥塞时给包打标记,让发端软减速)。

小数字:IB NDR 延迟 ~1 μs,调好的 RoCE ~2–2.5 μs——差 1–1.5 μs。对大批量训练 all-reduce(消息大、带宽主导),这点延迟差**对训练时间几乎不可见**;实测 256–1024 卡规模,配置正确的 RoCE 能拿到 IB **85–95%** 的训练吞吐。但 RoCE 若开了全局 PFC,暂停风暴会让以太网看起来比 IB 慢 **5–10×**——分水岭是运维水平,不是协议本身。

$$
t_{\text{net}}=\alpha_{\text{link}}+\frac{M}{\text{BW}},\quad
\alpha_{\text{IB}}\approx1\,\mu s,\ \alpha_{\text{RoCE}}\approx2\text{–}2.5\,\mu s
$$

(大 $M$ 时 $\tfrac{M}{\text{BW}}$ 主导,$\alpha$ 差被淹没→训练吞吐接近;小 $M$/延迟敏感场景 IB 占优。)

![[net-IB-vs-RoCE对比.png]]

![[net-069IB-RoCE矩阵.png]]

![[net-069无损流控机制.png]]

![[net-pfc-vs-ecn.png]]

```bash
# 确认 RDMA 设备与链路类型
ibstat                      # IB:State Active、Rate 400(NDR)
show_gids                   # RoCE:看 RoCE v2 GID 是否就绪

# RoCE 无损三件套(交换机侧,配错就是性能灾难)
#  1) PFC 只开 RDMA 优先级(如 priority 3),别开全局
#  2) ECN 标记阈值:AI 流量约 marking 500KB / drop 1MB
#  3) DCQCN 拥塞控制启用
# NCCL 选网卡:
export NCCL_IB_HCA=mlx5     # 指定走 IB/RoCE HCA
export NCCL_IB_GID_INDEX=3  # RoCE v2 常用
```

```text
❌ RoCE 上开全局 PFC（所有流量类都能被暂停）
   → 一处拥塞触发级联暂停风暴,整网慢 5–10×,看起来"以太网不行"
❌ 以为 IB 比 RoCE 快很多所以训练必须上 IB
   → 大消息训练吞吐两者差 5–15%,多数 workload 不可见;成本/运维才是决定项
✅ RoCE 只在 RDMA 优先级开 PFC + 配好 ECN/DCQCN;延迟极敏感或要 SHARP 才上 IB
```

## 面试高频
- **IB 和 RoCE 本质区别?** 都跑 RDMA;IB 信用流控天生无损、专网;RoCE 把 RDMA 封进以太网 UDP/IP,靠 PFC+ECN 人工凑无损。
- **RoCE 为什么难调?** 以太本会丢包,PFC 硬防丢但会头阻塞、配错引发全网暂停风暴;ECN 软减速;两者都要精调阈值。
- **延迟差多少、重要吗?** IB ~1μs vs RoCE ~2–2.5μs;大消息训练里这点差几乎不可见,小消息/延迟敏感场景 IB 占优。
- **什么时候选 IB?** 顶级集群、延迟极敏感、要 SHARP 网内归约、预算够、能接受单一供应商(DGX SuperPOD 默认 IB)。
- **什么时候选 RoCE?** 成本敏感、已有以太生态与运维、256–1024 卡规模——能拿 IB 85–95% 吞吐。
- **GPUDirect RDMA 是什么?** 网卡直接读写 GPU 显存、绕开 CPU 拷贝,IB/RoCE 都靠它。

## 关键事实
- IB NDR 约 1 μs 延迟,调好的 RoCE 约 1.5–2.5 μs;多数 AI 训练 workload 延迟差不可见(2025 行业对比)。
- 正确配置的 RoCE v2 可达 IB **85–95%** 训练吞吐(256–1024 GPU tier 2/3 部署,**2025**)。
- RoCE 无损关键:PFC 只开 RDMA 优先级 + 显式禁全局 PFC,ECN 标记约 500KB、丢弃约 1MB;全局 PFC 会引发级联暂停风暴致慢 5–10×(**2025**)。
- IB 用信用流控天生无损;RoCE v2 把 IB 语义封进 UDP/IP 跑标准以太,需独立 fabric 的是 IB(专用 HCA/交换机/线缆)。
- DGX SuperPOD 参考架构默认三层 fat-tree + Quantum-2 IB 400 Gb/s(NVIDIA,**2025**)。
