[[070 ZeRO 与 FSDP|ZeRO 与 FSDP]] 是改良的数据并行:普通 [[069 数据并行与 AllReduce|DP]] 让每张卡都存一份完整的**优化器态、梯度、参数**,极度冗余;ZeRO(Zero Redundancy Optimizer)把这三样**逐级切到各卡**(Stage 1/2/3),用通信换显存,让放不下的模型也能训。PyTorch 的 **FSDP** 是 ZeRO-3 的等价实现。

## 直觉

回看 [[069 数据并行与 AllReduce|数据并行]]:N 张卡,每张都放**完整一份**模型。但训练时单卡显存装的远不止参数——用 Adam + 混合精度时,一个参数要占约 **16 字节**(下面拆),而其中绝大部分(fp32 参数副本、动量、方差)是优化器的"账本"。N 张卡存 N 份完全相同的账本,**纯冗余**。

ZeRO 的洞察:**这些东西不需要每卡都存一份完整的**。既然 DP 里各卡本来就要靠通信同步,不如索性把账本也**切开**:每卡只存 1/N,需要时临时通信凑齐。这样单卡显存随卡数**近线性下降**,而 DP 的"切 batch、各算各的"血统不变——不像 [[071 张量并行(Megatron)|TP]]/[[072 流水线并行与气泡|PP]] 要改计算图。

ZeRO 分三阶段,**切得越多省得越狠,通信也越多**:

1. **ZeRO-1**:切**优化器态**(最大头),几乎不增通信。
2. **ZeRO-2**:再切**梯度**,通信仍同 DP。
3. **ZeRO-3**:连**参数**也切,显存随 N 线性降到底,代价是通信约 1.5×。

**FSDP = ZeRO-3**:平时每卡只存参数分片,算某层前 all-gather 凑齐完整权重、算完立刻丢。

![[dist-ZeRO三阶段.png]]

## 例子

**先拆"16 字节/参数"这笔账**(混合精度 + Adam,$\Psi$=参数量):

| 项 | 精度 | 字节/参数 |
|---|---|---|
| fp16 参数(算用) | fp16 | 2 |
| fp16 梯度 | fp16 | 2 |
| fp32 参数副本(主权重) | fp32 | 4 |
| Adam 一阶动量 $m$ | fp32 | 4 |
| Adam 二阶动量 $v$ | fp32 | 4 |
| **合计** | | **16** |

后 12 字节(fp32 副本 + $m$ + $v$)是**优化器态**——最大头。详见 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]]。

**7.5B 模型、N=64 卡**(经典 ZeRO 论文数字):
- **纯 DP**:每卡 $16\Psi=16\times7.5\text{B}=120$ GB——单卡 80GB 装不下。
- **ZeRO-1**(切优化器态):$\approx 4\Psi+\frac{12\Psi}{N}=30+\frac{90}{64}\approx 31.4$ GB。
- **ZeRO-2**(再切梯度):$\approx 2\Psi+\frac{14\Psi}{N}=15+1.6\approx 16.6$ GB。
- **ZeRO-3**(再切参数):$\approx \frac{16\Psi}{N}=\frac{120}{64}\approx 1.9$ GB。

ZeRO-3 把 120GB 砍到不到 2GB——这就是它能训万亿参数的原因。代价:ZeRO-3 通信量约为纯 DP 的 **1.5 倍**(每层前 all-gather 参数 + 反向 reduce-scatter 梯度)。

![[dist-FSDP流程.png]]

## 原理

**1. 显存公式。** 设参数量 $\Psi$、数据并行度 $N$、每参数总状态 $K=16$ 字节(fp16 参/梯各 2 + fp32 副本/动量/方差共 12):

$$
\text{DP}: K\Psi,\quad
\text{ZeRO-1}: 2\Psi+2\Psi+\frac{12\Psi}{N},\quad
\text{ZeRO-2}: 2\Psi+\frac{2\Psi+12\Psi}{N},\quad
\text{ZeRO-3}: \frac{K\Psi}{N}.
$$

(此处只算"模型状态",不含激活。激活另外由 [[065 梯度检查点与激活重计算|激活重计算]]、序列并行管。)只有 ZeRO-3 让**参数**也分片,显存才真正随 $N$ 线性降到底。

**2. ZeRO-3 / FSDP 的执行流(关键)。** 平时每卡只持有每层参数的 $1/N$ 分片:
- **前向到某层**:`all-gather` 把该层 $N$ 个分片凑成完整权重 → 算这一层 → **算完立刻丢弃**非本卡分片,释放显存。峰值显存 ≈ 常驻分片 + **当前单层**完整权重,而非整模型。
- **反向到某层**:同样 all-gather 凑齐权重算梯度,梯度算完用 `reduce-scatter` 把全局梯度的 $1/N$ 留给本卡(对应它持有的参数分片)。
- **更新**:每卡只更新自己那 $1/N$ 的参数 + 优化器态。无冗余。

所以 ZeRO-3 = DP 的"切 batch"+"逐层 all-gather/reduce-scatter 分片存储"。它不切计算图,**仍是数据并行**。

**3. 为什么通信只多 1.5×。** 纯 DP 每步 1 次梯度 all-reduce(= reduce-scatter + all-gather,通信量 $2\Psi$)。ZeRO-3 把梯度 all-reduce 拆成反向的 reduce-scatter($\Psi$),再加每层前 all-gather 参数($\Psi$),总通信约 $3\Psi$ vs DP 的 $2\Psi$——即 1.5×。**用 50% 额外通信换显存随 N 线性下降**,大多数场景非常划算。

**4. 与 TP/PP 的本质区别。** ZeRO/FSDP **只切存储,不切计算**:每卡仍跑完整的前向/反向(算之前临时凑齐权重)。TP 切的是单层矩阵乘本身(计算被分摊)、PP 切层(计算被分到不同 stage)。所以 ZeRO 实现简单、无需改模型代码,常作并行**基座**,再叠 TP/PP(见 [[068 并行总览：DP、TP、PP、EP、SP|并行总览]])。

**5. Offload:CPU/NVMe 接力。** ZeRO-Offload 把优化器态和优化器计算搬到 **CPU 内存**;ZeRO-Infinity 进一步用 **NVMe** SSD,让单卡甚至能训百亿/千亿模型(以吞吐为代价)。

## 三个 stage 的通信量逐项对账(为什么只 ZeRO-3 多通信)

把 [[074 通信原语与计算通信重叠|原语]] 算清楚,就明白"省显存"和"加通信"的兑换率。设参数量 $\Psi$,以一步训练为单位,只数模型状态相关的通信(单位:参数个数,实际字节再乘精度)。

| | 优化器态 | 梯度 | 参数 | 通信原语 | 总通信量 |
|---|---|---|---|---|---|
| **纯 DP** | 全副本 | all-reduce | 全副本 | 1×all-reduce($2\Psi$) | $2\Psi$ |
| **ZeRO-1** | 切 | all-reduce | 全副本 | 同 DP | $2\Psi$ |
| **ZeRO-2** | 切 | reduce-scatter | 全副本 | reduce-scatter($\Psi$)+更新后无需 all-gather参数(参数没切) | $2\Psi$ |
| **ZeRO-3** | 切 | reduce-scatter | 切 | 前向 all-gather 参数($\Psi$)+反向 all-gather 参数($\Psi$)+reduce-scatter 梯度($\Psi$) | $3\Psi$ |

要点:**ZeRO-1/2 通信量和纯 DP 一样($2\Psi$)**——它们只是把"参数服务器式的存储冗余"切掉,而梯度同步该传的还得传(all-reduce = reduce-scatter + all-gather,字节数不变)。**只有 ZeRO-3 因为参数也分片,每次用参数前都要临时 all-gather 凑齐(前向一次、反向一次)**,才多出那 $\Psi$,总 $3\Psi$ = DP 的 1.5×。所以**显存够用时优先 ZeRO-2**(免费切掉优化器态+梯度冗余,零通信代价);单卡连参数都装不下,才上 ZeRO-3 付那 50% 通信税。

## bf16 训练:16 字节还是 18 字节?

经典 ZeRO 论文按 **fp16 混合精度**算出 16 字节/参数(fp16 参 2 + fp16 梯 2 + fp32 副本 4 + $m$ 4 + $v$ 4)。现代大模型多用 **bf16**(见 [[064 混合精度 FP16、BF16 与 FP8]]),口径有两种,面试容易被追问:

- **梯度也存 fp32**(很多框架默认):bf16 参 2 + **fp32 梯 4** + fp32 副本 4 + $m$ 4 + $v$ 4 = **18 字节/参数**。
- **梯度 bf16**(更省):bf16 参 2 + bf16 梯 2 + fp32 副本 4 + $m$ 4 + $v$ 4 = **16 字节/参数**。

7B 模型按 18 字节算就是 $7\times18=126$ GB(不是 112)。**记住分解过程而非死记 16**——被问"bf16 下为什么是 18"能拆出来,比背数字强。注意:bf16 比 fp16 动态范围大(8 位指数,同 fp32),通常不需要 loss scaling,但精度位少 → 主权重仍用 fp32 防更新被舍入。

## ZeRO-3 / FSDP 的工程优化:prefetch 与通信重叠

ZeRO-3 朴素实现"算这层前才 all-gather"会让 GPU **干等参数到位**。生产实现靠两招把通信藏进 [[074 通信原语与计算通信重叠|计算的影子]]:
- **前向 prefetch**:算第 $i$ 层时,后台流已经在 all-gather 第 $i{+}1$ 层的参数;算完第 $i$ 层、丢弃其参数,第 $i{+}1$ 层参数刚好就位。
- **反向 prefetch**:反向逐层回传,提前 all-gather 上一层(更靠前)参数;梯度 reduce-scatter 也与反向计算重叠。
- **ZeRO++**(Wang 2023,arXiv:2306.10209):权重量化到 int8 做 all-gather(qwZ)、分层分片(hpZ,机内全量参数避免跨机 all-gather)、梯度量化(qgZ),把 ZeRO-3 通信再砍约 4×,专治带宽差的集群。

**FSDP 的 `reshard_after_forward`**:控制前向用完参数后是否立即重新分片。设 `True` 最省显存(立即丢)、`False` 用显存换通信(前向后不丢、反向不必再 all-gather)——典型在反向紧接的层上权衡。

## ZeRO/FSDP 与 TP/PP 的叠加(3D/4D 并行里的角色)

ZeRO 是**数据并行维度**的优化,可与切计算的并行正交叠加:
- **FSDP + TP**:TP 切机内单层矩阵乘,FSDP 在 TP 组**之间**切优化器态/梯度。注意 ZeRO-3 全分片 + TP 会重复切参数、通信叠加,工程上常用 **ZeRO-1 + TP + PP**(Megatron-DeepSpeed 的 3D 配方),让 ZeRO 只切优化器态、把切参数的活交给 TP/PP。
- **HSDP(Hybrid Sharded Data Parallel)**:机内 ZeRO-3 全分片、机间只做 DP 复制——把昂贵的跨机 all-gather 限制在机内 NVLink,机间只走轻量梯度同步。大集群(几百上千卡)常用,避免 ZeRO-3 的 all-gather 打满跨机网络。
- **MoE + ZeRO**:专家参数极多但每 token 只激活少数(见 [[049 专家并行 EP 与 MoE 部署]]),ZeRO 切非专家参数、EP 切专家,二者配合。

## 代码

```python
# ❌ 普通 DP：每卡都存完整优化器态（fp32 副本+m+v）→ 12 字节/参数 ×N 份冗余
model = DDP(model)            # 7.5B 模型每卡需 120GB，80GB 卡放不下

# ✅ ZeRO（DeepSpeed）：一个配置切掉冗余，stage 越高省越多
ds_config = {
    "train_micro_batch_size_per_gpu": 4,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,                       # 1=切优化器态 2=+梯度 3=+参数
        "offload_optimizer": {"device": "cpu"},   # ZeRO-Offload：优化器态搬 CPU
        "offload_param":     {"device": "cpu"},   # 参数也可卸载（更省、更慢）
    },
}
import deepspeed
engine, opt, _, _ = deepspeed.initialize(model=model, config=ds_config)
for batch in loader:
    loss = engine(batch).loss
    engine.backward(loss)     # 反向中 reduce-scatter 梯度，更新只动本卡分片
    engine.step()
```

```python
# PyTorch 原生 FSDP（≡ ZeRO-3）：按层包成可分片单元
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

model = FSDP(
    model,
    auto_wrap_policy=functools.partial(            # 每个 transformer block 单独分片
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    ),
    # 算这层前 all-gather 凑齐权重 → 算完即丢；反向 reduce-scatter 梯度
)
# 训练循环和普通模型一样：loss.backward(); opt.step()
# ❌ 坑：auto_wrap 粒度太粗（整模型一个单元）→ all-gather 一次凑齐全部，等于没分片
# ✅ 对：按 transformer 层分片，峰值显存 ≈ 单层完整大小
```

## 面试高频

- **Q:ZeRO 三个 stage 分别切什么?省多少?** A:Stage 1 切优化器态(最大头,fp32 副本+动量+方差=12字节/参);Stage 2 再切梯度;Stage 3 再切参数。单卡模型状态显存:DP 是 16Ψ,ZeRO-3 降到约 16Ψ/N,随卡数近线性下降。
- **Q:ZeRO 和数据并行什么关系?** A:ZeRO 是改良的数据并行——仍切 batch、各卡跑完整前向反向,只是把"每卡存完整优化器态/梯度/参数"的冗余切成 1/N 分片,需要时通信凑齐。血统是 DP,不切计算图。
- **Q:ZeRO-3 / FSDP 怎么执行?显存峰值多少?** A:平时每卡只存参数分片;算某层前 all-gather 凑齐完整权重,算完立即丢弃;反向 reduce-scatter 梯度。峰值显存≈常驻分片+当前单层完整权重,而非整模型。
- **Q:ZeRO-3 比普通 DP 多多少通信?值不值?** A:约 1.5×(梯度 reduce-scatter Ψ + 每层前 all-gather 参数 Ψ,共 3Ψ vs DP 的 2Ψ)。用 50% 额外通信换显存随 N 线性降,绝大多数场景划算。
- **Q:ZeRO 和张量并行有何区别?何时叠加?** A:ZeRO 只切存储不切计算(实现简单、无需改模型),TP 切单层矩阵乘的计算本身(限机内、要改模型)。模型放不下先上 ZeRO/FSDP;单层都超卡再叠 TP,层多再叠 PP。
- **Q:FSDP 和 DeepSpeed ZeRO 是一回事吗?** A:思想等价,FSDP≈ZeRO-3(参数/梯度/优化器态全分片)。FSDP 是 PyTorch 原生,DeepSpeed ZeRO 功能更全(Offload/Infinity 把状态卸到 CPU/NVMe)。
- **Q:ZeRO-1/2 为什么不增通信,只有 ZeRO-3 增?** A:ZeRO-1/2 只切存储,梯度同步该传的还传(all-reduce ≡ reduce-scatter + all-gather,字节不变),总量仍 $2\Psi$;ZeRO-3 因参数也分片,前向反向用参数前各要 all-gather 一次($\Psi+\Psi$)+ 梯度 reduce-scatter($\Psi$)= $3\Psi$,即 1.5×。
- **Q:bf16 训练每参数多少字节?** A:看梯度精度。梯度存 fp32 是 18 字节(2+4+4+4+4),梯度 bf16 是 16 字节(2+2+4+4+4)。主权重恒为 fp32 防小更新被舍入。
- **Q:大集群为什么用 HSDP 而非纯 ZeRO-3?** A:纯 ZeRO-3 的 all-gather 跨机会打满慢网。HSDP 机内全分片(走 NVLink)、机间只做 DP 梯度复制,把昂贵通信关在机内。
- **Q:ZeRO-3 怎么避免 GPU 等参数干等?** A:prefetch——算第 $i$ 层时后台 all-gather 第 $i{+}1$ 层参数,通信藏进计算;ZeRO++ 进一步量化通信量。
- **陷阱**:① ZeRO 不省激活显存(那靠重计算/序列并行);② FSDP 分片粒度太粗等于没分片;③ Offload 省显存但吞吐掉很多;④ Stage3 小模型时通信占比高,未必比 ZeRO-2 快;⑤ ZeRO-3 + TP 重复切参数,工程上常用 ZeRO-1 + TP + PP;⑥ 忘了 prefetch/重叠,ZeRO-3 会被 all-gather 延迟拖垮。

## 关键事实

- **ZeRO** 由 Rajbhandari 等(微软,2020,SC'20,arXiv:1910.02054)提出,Zero Redundancy Optimizer:分阶段切优化器态(P_os)、梯度(P_g)、参数(P_p),消除数据并行内存冗余,可训万亿参数模型;论文给出 16Ψ→16Ψ/N 的显存模型。
- **混合精度 + Adam 每参数约 16 字节**:fp16 参数 2 + fp16 梯度 2 + fp32 主权重 4 + Adam 一阶动量 4 + 二阶动量 4(见 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]])。
- **ZeRO-3 通信约为纯 DP 的 1.5×**(论文证明):reduce-scatter 梯度 + all-gather 参数,换取参数也分片、显存随 N 线性下降。
- **FSDP**(PyTorch,Zhao 等,2023,arXiv:2304.11277)是 ZeRO-3 的原生等价实现,按层 all-gather/reduce-scatter;已成 PyTorch 大模型训练标准路径之一。
- **ZeRO-Offload**(Ren 等,2021,arXiv:2101.06840)把优化器态/计算卸到 CPU;**ZeRO-Infinity**(Rajbhandari 等,2021,arXiv:2104.07857)进一步用 NVMe,单节点可训万亿参数。
- **ZeRO++**(Wang 等,2023,arXiv:2306.10209):权重量化 all-gather(qwZ)+ 分层分片(hpZ)+ 梯度量化(qgZ),把 ZeRO-3 通信量降约 4×,改善低带宽集群与小 batch 下的扩展性。
- **bf16 训练显存口径**:梯度存 fp32 时 18 字节/参数,梯度 bf16 时 16 字节/参数;bf16(8 位指数,范围同 fp32)通常免 loss scaling,但主权重仍 fp32(《Mixed Precision Training》Micikevicius 等,2017,arXiv:1710.03740)。
- **HSDP**(Hybrid Sharded Data Parallel,PyTorch FSDP 选项):机内全分片 + 机间 DP 复制,限制跨机 all-gather,大集群常用。
- 关联:基线 [[069 数据并行与 AllReduce|数据并行]];显存拆解 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]];激活由 [[065 梯度检查点与激活重计算|激活重计算]] 管;切计算的 [[071 张量并行(Megatron)|TP]]/[[072 流水线并行与气泡|PP]];全谱 [[068 并行总览：DP、TP、PP、EP、SP|并行总览]];优化器 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|Adam]]。
