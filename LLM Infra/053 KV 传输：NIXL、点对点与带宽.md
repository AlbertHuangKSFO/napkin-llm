[[053 KV 传输：NIXL、点对点与带宽|KV 传输]]是 [[048 为何分离 prefill 与 decode|PD 分离]] 的命门:[[013 Prefill 阶段：计算受限|Prefill]] 卡算出整条 prompt 的 [[LLM/102 KV-Cache|KV-Cache]] 后,必须把它搬到 [[014 Decode 阶段：访存受限|Decode]] 卡才能续生成。**NIXL**(NVIDIA Inference Xfer Library,GTC 2025 开源,[[052 NVIDIA Dynamo：分布式推理框架与 SLO Planner|Dynamo]] 的传输层)用**点对点 RDMA**把 KV 在卡间以接近线速搬运;支持 RDMA/InfiniBand、RoCE、TCP、NVMe-oF、S3 等多后端。关键权衡是**传 KV vs 重算 KV**:KV 大、网快则传输划算;网慢则 decode 卡重算 prefill 可能更优。带宽是这套架构能否成立的瓶颈。

## 直觉

prefill 算 KV 像后厨把一大锅料备好,decode 续生成像前厅接着上菜——但厨房和前厅在**不同楼**(不同 GPU/节点),那锅料(KV)得有人搬上去。搬运方式决定成败:用**点对点 RDMA**(GPU-Direct,绕过 CPU)像装了直达货梯,几毫秒送到;若只有普通网络(TCP 经 CPU 拷贝),就像用楼梯抬,长 prompt 的大锅 KV 会把传输堵成瓶颈,抵消分离的收益。

另一个反直觉:有时**不传反而更快**。如果网络慢、而 prompt 不长,让 decode 卡自己把 prompt 再 prefill 一遍(重算 KV),可能比等传输更省时——这就是"传输 vs 重算"的权衡。

## 例子

KV 传输量的带宽账(逐步算):

- 设 Llama-3-70B,80 层、KV 头 8、head_dim 128、BF16(2 字节),GQA。单 token 的 KV ≈ $2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ 字节 ≈ **0.31 MB/token**(K 和 V 各一份故乘 2)。
- 一条 **8k token** 的 prompt:KV ≈ $8192 \times 0.31\,\text{MB} \approx 2.5\,\text{GB}$ 要从 prefill 传到 decode。
- 在 **400 Gb/s(50 GB/s)IB-NDR** 上:$2.5/50 = 50\,\text{ms}$;在 **InfiniBand HDR(200 Gb/s)** 上约 100 ms——若 TTFT SLO 是 200 ms,这笔传输不可忽视,必须高速网才稳。
- 反例(重算更优):一条 256 token 短 prompt,KV 仅 ~80 MB,但若只有 TCP/慢网,搬运 + 排队可能比 decode 卡花几毫秒重算还慢 → 选重算。
- NIXL 实测:InfiniBand HDR 上一个 47 token 短 prompt 的 KV 传输 **sub-5ms**;特定配置(Augmented Memory Grid)KV 投递 **最高 252 GB/s 每节点**。

## 原理

KV 传输时间由数据量与链路带宽决定;是否值得传,取决于它与**重算成本**的比较:

$$
T_{\text{transfer}} = \frac{S_{\text{kv}}}{B_{\text{link}}},\qquad
S_{\text{kv}} = 2\,L\,N_{\text{layers}}\,H_{\text{kv}}\,d_{\text{head}}\,b
$$

$$
\text{传 KV 划算} \iff
\underbrace{\frac{S_{\text{kv}}}{B_{\text{link}}}}_{\text{传输}}
\;<\;
\underbrace{\frac{2\,L\,N_{\text{params}}}{C_{\text{FLOPS}}}}_{\text{decode 卡重算 prefill}}
$$

左边随 **prompt 长度线性增**但被带宽除;右边是重算的算力代价。长 prompt + 快网 → 左小右大,传输胜;短 prompt + 慢网 → 重算胜。点对点 RDMA/GPU-Direct 的意义就是把 $B_{\text{link}}$ 顶到 HBM 量级、并绕开 CPU 拷贝,让左边尽量小。

## 图

![[disagg-KV传输路径.svg]]

![[disagg-053传输vs重算交叉.svg]]

![[disagg-053链路带宽账.svg]]

## 代码

vLLM/Dynamo 用 NIXL 连接器把 KV 从 producer 推到 consumer:

```python
# ❌ 慢路径:KV 经 CPU 中转 + TCP(host→host 拷贝),长 prompt 直接打满网络
# socket.send(kv.cpu().numpy().tobytes())   # 多次拷贝,延迟高

# ✅ NIXL 点对点:GPU-Direct RDMA,绕过 CPU,接近线速
from vllm.config import KVTransferConfig
producer = KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_producer")
consumer = KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_consumer")
# 后端按集群选:同节点 NVLink / 跨节点 RDMA-IB / 退化 TCP

# 传输 vs 重算的决策(示意)
def should_transfer(kv_bytes, link_GBps, prompt_len, params, tflops):
    t_transfer = kv_bytes / (link_GBps * 1e9)
    t_recompute = 2 * prompt_len * params / (tflops * 1e12)
    return t_transfer < t_recompute      # ✅ 短 prompt+慢网时可能选重算
```

`❌` 经 CPU 的 TCP 路径多次内存拷贝、带宽低,长 prompt 的大 KV 会成新瓶颈;`✅` NIXL 走 GPU-Direct RDMA 点对点,接近线速,并按 prompt 长度/网速在传输与重算间择优。

## 面试高频

- **Q:PD 分离里 KV 怎么从 prefill 到 decode?** prefill 算完整条 prompt 的 KV,经**点对点传输**(NIXL:RDMA/IB、RoCE、TCP、NVMe-oF)搬到 decode 卡;理想用 GPU-Direct RDMA 绕过 CPU。
- **Q:KV 传输为什么是瓶颈?** KV 量随 prompt 长度线性增,长 prompt 可达 GB 级;若链路带宽不足(无高速网),传输时间吃掉 TTFT 预算,抵消分离收益。
- **Q:什么时候宁可重算 KV 也不传?** 短 prompt(KV 小)+ 慢网时,decode 卡自己重 prefill 一遍可能比等传输更快;长 prompt + 快网则传输划算。
- **Q:NIXL 是什么?** NVIDIA Inference Xfer Library,GTC 2025 开源的点对点 KV 传输库,多后端(RDMA/IB、RoCE、TCP、NVMe-oF、S3),是 Dynamo/TRT-LLM/vLLM/SGLang 的传输层。
- **Q:怎么估一条 prompt 要传多少 KV?** $S_{\text{kv}}=2\,L\,N_{\text{layers}}\,H_{\text{kv}}\,d_{\text{head}}\,b$;再除以链路带宽得传输时间。GQA/MLA 减小 $H_{\text{kv}}$ 从而减小传输量。

## 关键事实

- **NIXL** = NVIDIA Inference Xfer Library,**GTC 2025** 开源;点对点 KV 传输,支持 **RDMA/IB、RoCE(UCX)、TCP、NVMe-oF、S3**;是 [[052 NVIDIA Dynamo：分布式推理框架与 SLO Planner|Dynamo]]、TensorRT-LLM、vLLM、SGLang、LMCache 的传输层。
- 量级:生产多节点推荐 **IB HDR/NDR(400+ Gb/s)**;47-token 短 prompt 传输 **sub-5ms**;特定配置 KV 投递 **最高 252 GB/s 每节点**。
- 核心权衡:**传 KV vs 重算 KV**——长 prompt + 快网 → 传输胜;短 prompt + 慢网 → 重算胜。GQA/MLA 减少 KV 体积,直接减小传输压力。
