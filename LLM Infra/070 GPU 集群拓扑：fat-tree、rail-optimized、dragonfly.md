这是 [[LLM Infra]] 域讲**集群网络怎么布线**,承接 [[009 节点与机架：PCIe、GB200 NVL72、scale-up 与 scale-out|scale-up/out]] 的 scale-out 物理结构,决定 [[069 InfiniBand vs RoCE|IB/RoCE]] 链路怎么组网、[[066 NCCL：集合通信库与原语|NCCL]] 怎么选路,参见 [[LLM/069 数据并行与 AllReduce|AllReduce]]。一句话:**GPU 集群主流是 fat-tree(多层 Clos、无阻塞,任意两卡带宽稳定);rail-optimized 在其上叠一层巧思——把各节点的同号 GPU 连到同一个 rail 交换机,让大多数集合通信一跳直达不挤上层,跨 rail 流量由 PXN 先经机内 NVLink 挪到同号卡再发;dragonfly 则用分组密连/稀连省交换机,适合超大规模但路径不均。**

类比:fat-tree 像有多条等宽主干道的城市路网,从任意小区到任意小区都不堵(无阻塞)、但要花很多路。rail-optimized 像给"同工种"的人安排了直达班车:所有节点的 0 号工位连 0 号班车线、1 号连 1 号线——同工位之间(数据并行里最常见的同号 GPU 通信)上车就到,不用换乘进市中心(spine)。PXN 是"换乘技巧":你要去别条线的站,先在自家楼里(NVLink)走到对应工位,再坐那条线的直达班车。dragonfly 像把城市分成几个"组团",组内路修得密、组间只架几座桥——省钱,但跨组要绕桥。

为什么 rail-optimized 有效?LLM 训练/推理里,数据并行的 all-reduce、流水并行的相邻 stage、MoE 的部分通信,**大量发生在不同节点的同号 GPU 之间**。把同号 GPU 全挂到同一个 rail 交换机,这类流量就只在该 rail 内一跳完成,根本不上第二层 spine——上层带宽需求骤降,可少买 spine 交换机。**PXN**(PCIe×NVLink)补全跨 rail 那部分:目标在别的 rail 时,先机内 NVLink 把数据挪到本节点对应 rail 的那张卡,再走目标 rail 一跳直达,并能把多条消息聚合(最多 8 条合 1 条),降 spine 流量、提消息率。

小数字:8 GPU/节点的标准配置,8 个 rail。256 节点 = 2048 GPU,每个 rail 交换机接 256 张同号卡。一次数据并行 all-reduce 若全在同号卡间,**0% 流量进 spine**;若是任意全交换(all-to-all),约 $7/8$ 的目标在异 rail,PXN 把它们先 NVLink 归位再发——把"跨 rail 的随机流量"转成"同 rail 的规整流量"。fat-tree 无阻塞要求每层上下行带宽相等(收敛比 1:1),这是它贵但稳的根源。

$$
\text{fat-tree 无阻塞}:\ \text{上行带宽}=\text{下行带宽}\ (\text{收敛比}\ 1{:}1),\qquad
\text{rail-opt 上层流量}\downarrow\ \text{(同号 GPU 同 rail → 不进 spine)}
$$

(dragonfly 用更少交换机换"非全无阻塞"和路径长度不均;超大规模才划算。)

![[net-rail-optimized拓扑.svg]]

![[net-070三拓扑架构.svg]]

![[net-070PXN跨rail.svg]]

```bash
# 让 NCCL 用上 rail/PXN 拓扑感知(通常默认开)
export NCCL_PXN_DISABLE=0          # 启用 PXN(跨 rail 经 NVLink 中转)
export NCCL_P2P_LEVEL=NVL          # 机内走 NVLink
export NCCL_NET_GDR_LEVEL=PXB      # GPUDirect RDMA 等级
# 确认网卡-GPU 亲和(同号 GPU 应连同 rail NIC)
nvidia-smi topo -m                 # 看 GPU↔NIC 的 PXB/NVLink 关系
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH ./all_reduce_perf -b 1G -e 1G -g 8
```

```text
❌ 任意把 GPU↔NIC 乱接,不按 rail 对齐
   → 大量集合通信被迫上 spine,上层成瓶颈,白买的 fat-tree 带宽用不上
❌ 关掉 PXN,跨 rail 流量直接打 spine
   → spine 拥塞、消息率低,all-to-all/all-reduce 变慢
✅ 同号 GPU 对齐到同 rail + 开 PXN;用 nvidia-smi topo -m 验证 GPU-NIC 亲和
```

## 面试高频
- **为什么 GPU 集群普遍用 fat-tree?** 多层 Clos 无阻塞,任意 GPU 对带宽稳定;分布式训练通信模式动态变化,需要这种可预测性。
- **rail-optimized 解决什么?** 把各节点同号 GPU 连同一 rail,使常见的同号 GPU 集合通信一跳直达、不挤 spine,降上层带宽需求/成本。
- **PXN 是什么、怎么工作?** 跨 rail 时先用机内 NVLink 把数据挪到本节点对应 rail 的 GPU,再走该 rail 一跳直达 spine 之下;还能聚合最多 8 条消息。
- **dragonfly 适合什么?** 超大规模:分组密连/组间稀连,省交换机与线缆,代价是非全无阻塞、路径长度不均。
- **fat-tree 无阻塞的条件?** 每层上下行带宽相等(收敛比 1:1),这是它带宽稳但成本高的根源。

## 关键事实
- DGX SuperPOD 用三层 fat-tree + Quantum-2 IB 400 Gb/s/端口,fat-tree 因任意 GPU 对带宽可预测而成 GPU 集群主流(NVIDIA,**2025**)。
- PXN 借 NVSwitch 先把数据移到与目标同 rail 的 GPU,再发出去不跨 rail;并聚合最多 8 条消息为 1 条、降 spine 流量(NVIDIA,**2025**)。
- dragonfly:交换机分组,组内密连、组间稀连,相比 fat-tree 减少交换机数,路径长度仍合理(行业架构综述,**2025**)。
- receiver-side PXN(NCCL Roadmap Q4 **2025**)在规划中,改进 all-to-all 收端瓶颈与负载不均。
