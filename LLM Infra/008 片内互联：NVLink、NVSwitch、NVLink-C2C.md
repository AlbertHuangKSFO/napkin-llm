[[008 片内互联：NVLink、NVSwitch、NVLink-C2C|片内互联]] 指 GPU 之间(及 GPU-CPU 之间)的高带宽直连——NVLink 是 GPU 对 GPU 的专用链路,NVSwitch 是把它扩成无阻塞全互联的交换芯片,NVLink-C2C 则把 CPU 与 GPU 在封装级用同一致缓存协议连成一体。

## 直觉:厂房内部的传送带 vs 厂区之间的卡车
PCIe(见 [[009 节点与机架：PCIe、GB200 NVL72、scale-up 与 scale-out|scale-up 与 scale-out]])像厂区门口那条限速的公路;**NVLink** 是车间内部高速传送带,直接把一台机床的产出甩给隔壁机床。当机床多到「两两都要直连」就连不过来了,于是上 **NVSwitch**——一个中央分拣中心,任意两台机床都能满速对接,不挑组合。**NVLink-C2C** 更进一步:把 CPU 和 GPU 焊在同一块「主板」上共享同一仓库(统一内存),不再来回搬货。

## 小数字例子:同样搬 1GB 激活
跨卡 all-reduce 一份 1GB 激活张量:
- **NVLink 4(H100,900 GB/s)**:≈ 1/900 ≈ **1.1 ms**。
- **NVLink 5(B200,1.8 TB/s)**:≈ **0.56 ms**。
- **PCIe Gen5(~128 GB/s)**:≈ **7.8 ms** —— 慢 7 倍。
推理时一个 70B 模型几十层、每层每 token 都要同步;若走 PCIe,通信时间会把 decode 拖垮。这就是为什么 [[004 算力 vs 带宽：Roofline 与算术强度|Roofline]] 里通信也算「带宽限」的一环。

## 原理:张量并行的通信量
张量并行(TP)度为 $p$、隐藏维 $d$、batch×seq 为 $T$,每层一次 all-reduce 的通信量约:

$$\text{Bytes} \approx 2 \cdot T \cdot d \cdot \text{(bytes/elem)} \cdot \frac{p-1}{p}$$

关键不是总量大,而是**它在每层、每 token 的关键路径上**,且无法被计算掩盖。所以 TP 的有效带宽必须 ≥ 计算速度配比,否则 GPU 等通信。这把 TP 牢牢锁进 NVLink 域:跨 PCIe 或网络做 TP,带宽掉一个数量级,直接不可行。

## 图
![[hw-NVLink互联拓扑.svg]]

![[hw-008互联搬1GB耗时.svg]]

## NVLink 代际与互联类型(已验证,2025–2026)

| 互联 | 带宽 | 用途 |
|---|---|---|
| NVLink 3(A100) | 600 GB/s/卡 | GPU↔GPU |
| NVLink 4(H100 / H200) | 900 GB/s/卡 | GPU↔GPU(单链 25 GB/s × 18) |
| NVLink 5(B200) | 1.8 TB/s/卡 | GPU↔GPU(单链 50 GB/s × 18) |
| NVSwitch(Hopper) | 每卡全互联 900 GB/s,4 颗/8 卡节点 | 无阻塞 all-to-all |
| NVSwitch(Blackwell,4th gen) | 72 端口/颗,144 端口交换、14.4 TB/s 无阻塞 | 机架级全互联(见 GB200 NVL72) |
| NVLink-C2C(Grace-Hopper / Grace-Blackwell) | 900 GB/s | CPU↔GPU 缓存一致互联,~7× PCIe Gen5 |

## 代码/配置:并行策略与互联绑定

```python
# ❌ 不看互联拓扑,把 TP=8 跨两台 4 卡 PCIe 机器拆 → 通信跨 PCIe/网络,吞吐崩
# tensor_parallel_size=8 但实际只有 4 卡走 NVLink,另 4 卡走慢路

# ✅ TP 限制在单个 NVLink/NVSwitch 域内(通常 = 单节点 8 卡)
# vLLM 例:
llm = LLM(
    model="meta-llama/Llama-3-70b",
    tensor_parallel_size=8,      # 8 卡都在同一 NVSwitch 全互联域 → all-reduce 走 NVLink
    pipeline_parallel_size=2,    # 跨节点用流水线并行(PP),通信稀疏、容忍慢链路
)
# 经验法则:NVLink 域内放 TP(频繁同步),域间放 PP/DP(稀疏通信)
```

```bash
# ✅ 查实际拓扑:NV# 表示 NVLink 直连,PIX/PHB 表示走 PCIe(慢)
nvidia-smi topo -m
```


![[hw-008TP与PP通信对比.svg]]

## 面试高频
- **NVLink、NVSwitch、PCIe 各是什么、差多少?** NVLink = GPU 间专用高带宽链路(900 GB/s~1.8 TB/s);NVSwitch = 把 NVLink 扩成无阻塞全互联的交换芯片;PCIe ≈ 128 GB/s 是数量级更慢的通用总线。差 7~14 倍。
- **为什么张量并行必须在 NVLink 域内?** TP 每层每 token 都要 all-reduce 激活,通信在关键路径且无法被计算掩盖;NVLink 带宽够才不让 GPU 空等。跨 PCIe/网络做 TP 带宽掉一个量级,不可行。
- **NVLink-C2C 解决什么?** CPU↔GPU 缓存一致统一内存,900 GB/s ≈ 7× PCIe Gen5;让 GPU 直接高速访问 CPU 侧大内存(放 KV-Cache 溢出、参数 offload),无 PCIe 拷贝开销。
- **8 卡节点里 NVSwitch 为什么是 4 颗?** Hopper 单 NVSwitch 端口有限,4 颗才能让 8 张卡两两都拿到满 900 GB/s 的无阻塞带宽。
- **TP vs PP 怎么按互联放?** NVLink 域内放 TP(频繁同步),域间用 PP/DP(通信稀疏,容忍 IB/以太低带宽)。

## 关键事实(2025–2026)
- **NVLink 代际带宽**:A100 = 600 GB/s,H100/H200 = 900 GB/s(NVLink 4,18×25 GB/s),B200 = 1.8 TB/s(NVLink 5,18×50 GB/s)。
- **NVSwitch**:Hopper 8 卡节点 4 颗交换芯片,每卡满 900 GB/s 全互联;Blackwell 第四代 72 端口/颗,NVLink 5 交换 144 端口、14.4 TB/s 无阻塞。
- **NVLink-C2C**:Grace-Hopper / Grace-Blackwell 封装级 900 GB/s,缓存一致,约 PCIe Gen5(128 GB/s)的 7 倍。
- **核心结论**:TP 锁在 NVLink/NVSwitch 域内;跨域(PCIe / IB / 以太)只放通信稀疏的 PP/DP。互联带宽直接决定并行策略边界。
