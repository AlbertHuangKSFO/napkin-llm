[[115 多 LoRA 服务：S-LoRA、Punica 与热插拔]] 是**在同一份基座权重上同时挂载成百上千个 [[LLM/091 高效微调：LoRA、QLoRA、Adapter、Prefix|LoRA]] adapter,按每条请求选择各自的 adapter,并在一个 batch 内用融合内核一起算**的服务范式。核心动机:LoRA adapter 只有几 MB(基座几十 GB),为每个微调版本各起一个副本极其浪费;**多租户**场景(一个产品给上千客户各训一个小 adapter)更需要单卡承载海量 adapter。两大技术:**S-LoRA** 的 **unified paging**(统一显存池管 adapter 权重 + KV)与 **Punica** 的 **SGMV** 融合内核(分段 gather 矩阵乘,异构 rank 同批计算)。运维难点是 **热插拔**:动态加载/卸载 adapter、冷启延迟、引用计数、长尾缓存。它建立在 [[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention]] 的分页思想上,并和 [[040 静态批、动态批与连续批|连续批]] 调度耦合。

## 直觉类比
基座是一家**中央厨房**(贵、只建一座),adapter 是一沓沓**小配方卡**(便宜、上千张)。传统做法是「每个配方各建一座厨房」——荒谬。多 LoRA 服务是:**一座厨房,来单时抽出对应配方卡照着微调口味**。SGMV 像一条流水线:同一批订单里有人要 A 配方、有人要 B 配方,厨师不是一份份做,而是把「按配方分段」的活儿塞进**一道工序**批量完成。Unified paging 像把配方卡和食材放进**同一个货架系统**按需取放,冷门配方暂存仓库(CPU),热门的放灶台边(GPU)。

## 小数字例子
基座 7B,FP16 约 14 GB,只载一份。每个 LoRA(rank=16)约 **几十 MB**。
- 一张 80 GB 卡:基座 14 GB + KV 预算约 50 GB,剩余可放 **上千个** adapter 权重(冷的还能换出 CPU)。
- 一个 batch 32 条请求,可能命中 **8 个不同 adapter**:SGMV 在**一个 kernel** 内按 adapter 分段,各取自己的 $A,B$ 矩阵批量算,而不是循环 8 次小 GEMM(否则 GPU 利用率极低)。
- S-LoRA 报告:相比逐个切换权重的朴素做法 / vLLM 早期,吞吐最高约 **4×**,可服务的 adapter 数提升数个数量级。

## 原理:SGMV 与 unified paging
LoRA 把权重更新写成低秩:$W' = W + \frac{\alpha}{r} BA$,其中 $A\in\mathbb{R}^{r\times d},\,B\in\mathbb{R}^{d\times r}$,秩 $r\ll d$。推理时不必合并,分两路:

$$
y \;=\; \underbrace{Wx}_{\text{所有请求共享的大 GEMM}} \;+\; \underbrace{\frac{\alpha}{r}\,B\,(A\,x)}_{\text{各请求自己的 adapter}}
$$

基座那项整批一起算(高效)。难点在第二项:batch 内每条请求用**不同的 $A,B$**(甚至不同 rank)。**SGMV(Segmented Gather Matrix-Vector multiplication)** 用一个 CUDA kernel 按请求的 adapter id **分段**,gather 出各自的低秩矩阵并批量做乘加——把 $B$ 个零碎小乘法合成一次高效内核。下图对比「逐 adapter 循环」与「SGMV 分段融合」为何吞吐差几倍:

![[srv-115SGMV内核.png]]**Unified paging** 则把不同 rank 的 adapter 权重和不同长度的 KV-Cache 放进**同一个分页内存池**:消除碎片,允许冷 adapter 换出到 CPU、热的留 GPU,无需为每个 adapter 预留固定显存。

![[srv-多LoRA服务架构.png]]

unified paging 的分页池细节:adapter 页与 KV 页混排在同一物理页池,显存吃紧时优先把冷 adapter 页换出到 CPU、腾给热请求的 KV 页:

![[orch-115分页池换出.png]]

## 配置 / 代码
```python
# vLLM:启用多 LoRA,按请求指定 adapter
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_loras=8,          # 单 batch 内最多同时活跃的 adapter 数
    max_lora_rank=32,     # 支持的最大 rank(决定显存与 kernel)
    max_cpu_loras=64,     # 可在 CPU 缓存、按需换入的 adapter 数
)

sp = SamplingParams(temperature=0.7, max_tokens=128)

# 同一引擎、不同请求挂不同 adapter(热插拔:路径可指向新下载的 adapter)
out_a = llm.generate("法律合同摘要:...", sp,
    lora_request=LoRARequest("legal_v3", 1, "/adapters/legal_v3"))
out_b = llm.generate("医疗问答:...", sp,
    lora_request=LoRARequest("med_v2", 2, "/adapters/med_v2"))
out_base = llm.generate("普通闲聊", sp)   # 不传 → 纯基座
```

```text
❌ 每个 adapter 各起一个进程/各载一份基座 → 14GB×N 显存爆炸,绝大多数 adapter 极少被调却长期占卡
✅ 一份基座 + max_loras 控制活跃集 + unified paging 冷热换出 → 单卡服务上千 adapter,SGMV 同批融合保持高吞吐
```

## 面试高频
- **为什么不能给每个 LoRA 各起一个副本?** 基座几十 GB,adapter 才几 MB;N 个副本 = N 份基座显存,且长尾 adapter 闲置占卡。多 LoRA 把基座共享、adapter 当「可换页的小权重」。
- **SGMV 解决什么问题?** batch 内不同请求用不同 adapter,若逐个做小 GEMM,GPU 严重欠饱和;SGMV 用一个 kernel 按 adapter 分段批量算,异构 rank 也能同批。
- **unified paging 是什么?** 把 adapter 权重(不同 rank)和 KV-Cache(不同长度)放进同一分页池,消碎片、支持冷热换出,不为每个 adapter 预留固定显存。
- **热插拔的运维痛点?** 冷启延迟(从存储拉取 + H2D 拷贝)、卸载需引用计数避免删掉在用 adapter、长尾只缓存热集、版本灰度回滚、rank 不一需 kernel 支持。
- **多 LoRA 和连续批怎么配合?** adapter 选择是逐请求的;调度器把不同 adapter 的请求混进同一连续批,SGMV 在算子层按段处理,batch 组成可逐 step 变化。
- **基座项和 adapter 项怎么算?** $y=Wx+\frac{\alpha}{r}B(Ax)$,基座大 GEMM 整批共享,adapter 低秩项 per-request 用 SGMV 融合。

## 关键事实
- **S-LoRA**,Sheng et al.,arXiv **2311.03285**(2023,MLSys 2024):unified paging 统一管 adapter 权重 + KV;单卡/多卡服务**数千** concurrent adapter,相比基线吞吐最高约 **4×**,可服务 adapter 数提升数个数量级。
- **Punica**,Chen et al.(2023→MLSys 2024):提出 **SGMV(Segmented Gather Matrix-Vector multiplication)** CUDA kernel,把多 adapter 的低秩计算批成一个内核;并含调度器把请求路由到活跃 GPU、迁移合并。
- vLLM / SGLang(2025)内置多 LoRA:`enable_lora` + `max_loras` / `max_lora_rank` / `max_cpu_loras`;支持运行时按 `LoRARequest` 热挂载,冷 adapter CPU 缓存按需换入。
- 典型场景:多租户 SaaS(每客户一 adapter)、按任务切风格/领域、A/B 与灰度;基座共享是省显存关键。
- 与 [[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention]] 同源(分页消碎片),与 [[040 静态批、动态批与连续批|连续批]] 调度耦合。
