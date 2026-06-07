[[068 并行总览：DP、TP、PP、EP、SP|并行总览：DP、TP、PP、EP、SP]] 是分布式训练的"地图":当一张 GPU 装不下模型或算不动数据时,沿**五个不同维度**把工作切到多卡——DP 切样本、TP 切单层矩阵、PP 切层、EP 切专家、SP 切序列;它们彼此**正交、可叠加**成 3D/4D 并行。选哪种的核心权衡是**显存 vs 通信**。

## 直觉

训练一个大模型,单卡可能卡在两件事上:**显存不够**(模型 + 优化器态 + 激活塞不下,见 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]])或**算得太慢**(一卡跑一个 batch 太久)。多卡并行就是把"同一坨工作"沿某个维度切开,分给多张卡。关键是:**沿哪个维度切?**

把模型训练想成一个三维方块:**样本(batch)× 深度(层)× 宽度(隐藏维)**,再加上 MoE 的**专家维**和长序列的**序列维**。五种并行各切一个维度:

- **DP(数据并行)**:复制整个模型到每张卡,切 **batch**。每卡算不同样本,梯度 [[069 数据并行与 AllReduce|all-reduce]] 同步。最简单、扩展性最好,但每卡都要放下整模型。
- **TP(张量并行)**:切**单层内部的权重矩阵**(宽度/隐藏维)。一层的矩阵乘横/纵切到多卡,每层结束 all-reduce 拼回。能放下"单层都太大"的模型,但通信极重——限在机内 NVLink。
- **PP(流水线并行)**:切**层**(深度)。每卡放连续几层,激活像流水线一样点对点往下传。通信最轻(只在边界传激活),适合跨机;但有"气泡"空转。
- **EP(专家并行)**:切 **MoE 专家**(按专家编号)。只在 MoE 层用,token 经两次 all-to-all 飞到专家所在卡,见 [[049 专家并行 EP 与 MoE 部署|专家并行]]。
- **SP(序列并行)**:切**序列长度**维。一种是配 TP 切 LayerNorm/Dropout 的激活省显存,另一种是独立的上下文并行处理长序列,见 [[073 序列并行与上下文并行|序列并行]]。

![[dist-并行总览.png]]

## 例子

**怎么选?一个决策流程(小数字)**。手头一个 70B 模型,单卡 80GB:

1. **能塞进单卡吗?** 70B fp16 参数就 140GB,加梯度+Adam 优化器态(见下),总需约 70B×16 字节=**1120GB**,单卡远不够。
2. **先省冗余:ZeRO/FSDP**。纯 DP 每卡都存这 1120GB——纯浪费。[[070 ZeRO 与 FSDP|ZeRO]] 把优化器态/梯度/参数切到 N 卡,N=64 时每卡降到约 1120/64≈**17.5GB**,常常一步到位。
3. **单层都放不下?加 TP**。若隐藏维巨大、单层矩阵超卡,上张量并行,但 **TP 度数限在机内**(如 8 卡一机,走 NVLink),因为每层都要 all-reduce。
4. **层太多?加 PP 跨机**。深度方向切层,跨机用点对点传激活(便宜),配 1F1B 调度压气泡,见 [[072 流水线并行与气泡|流水线并行]]。
5. **是 MoE?加 EP**;**长上下文?加 SP/CP**。

**经典 3D 并行配置**:`DP × TP × PP`。比如 1024 卡 = `DP=16 × TP=8 × PP=8`。TP=8 限在机内 8 卡(NVLink),PP=8 跨 8 机,DP=16 再复制 16 份。Megatron-Turing NLG 530B 就是这种叠法。

![[dist-并行选型决策.png]]

## 原理

**1. 五种并行切的维度(正交性)。** 设全局 batch $B$、层数 $\ell$、隐藏维 $h$、专家数 $E$、序列长 $L$:

| 并行 | 切的维度 | 每卡持有 | 通信 | 何时用 |
|---|---|---|---|---|
| DP | batch $B$ | **整个**模型副本 | 梯度 all-reduce(每步1次) | 默认首选,模型放得下 |
| TP | 隐藏维 $h$(单层矩阵) | 每层的一**部分**权重 | 每层 all-reduce(前+后向各2次) | 单层太大,限机内 |
| PP | 层 $\ell$ | 连续**几层** | 边界点对点传激活 | 层多,可跨机 |
| EP | 专家 $E$ | 一**部分**专家 | 每 MoE 层 2 次 all-to-all | MoE 模型 |
| SP | 序列 $L$ | 序列的一**段** | all-gather/reduce-scatter 或环形 | 长上下文/省激活 |

它们正交,可同时用:总卡数 $=d_{\text{DP}}\cdot d_{\text{TP}}\cdot d_{\text{PP}}\cdot d_{\text{EP}}$。

**2. 显存 vs 通信的根本权衡。**
- **DP** 不省单卡模型显存(每卡全副本),但**几乎不增加额外通信复杂度**(梯度 all-reduce 可与反向重叠),扩展性最好。它解决的是"算得快",不是"放得下"。
- **TP** 把单层权重和激活都切了,**最省单层显存**,代价是**每层都要同步**——通信量正比于激活大小、频率极高,所以必须走机内高带宽 NVLink,跨机会被带宽掐死。
- **PP** 把层切开,**通信最省**(只在 stage 边界点对点传一次激活),适合跨机扩展;代价是流水线**气泡**(填充/排空阶段卡空转),占比 $\frac{p-1}{m+p-1}$($p$ 阶段、$m$ 微批),见 [[072 流水线并行与气泡|流水线气泡]]。

口诀:**TP 通信最重(限机内)→ EP 次重 → PP 最轻(可跨机)→ DP 可重叠、扩展最好**。

**3. ZeRO 与 DP 的关系。** [[070 ZeRO 与 FSDP|ZeRO]] 本质是**改良的 DP**:仍切 batch,但把"每卡都存一份优化器态/梯度/参数"的冗余**切掉**,用通信换显存。它让 DP 在不引入 TP/PP 复杂度的前提下,也能训放不下的模型——因此现代配方常以 ZeRO/FSDP 为基座,再叠 TP/PP。

**4. 通信原语是底座。** 所有并行都建在集合通信原语上:DP 用 all-reduce、TP 用 all-reduce/all-gather、PP 用 send/recv、EP 用 all-to-all、SP 用 reduce-scatter/all-gather。把它们与计算**重叠**是性能关键,见 [[074 通信原语与计算通信重叠|通信原语与重叠]]。

**5. Ring AllReduce 的通信量(为什么 DP 能扩展)。** $N$ 卡同步一个 $M$ 字节梯度,环形 all-reduce 分 reduce-scatter + all-gather 两阶段,每卡总收发约 $2M\cdot\frac{N-1}{N}\approx 2M$ 字节——**与卡数 $N$ 几乎无关**(常数!)。这就是 DP 扩展性最好的根本原因:加卡不增加每卡通信量。且这 $2M$ 的传输能与反向计算**重叠**(梯度算完一层就开始 all-reduce 该层),几乎隐藏掉。对比 TP 每层都要 all-reduce 激活、通信量随频率累积,这就是「TP 限机内、DP 可跨机扩展」的量化依据。

**6. 流水线气泡的数字。** $p$ 个 stage、$m$ 个 micro-batch,气泡占比 $\frac{p-1}{m+p-1}$。$p=8,m=8$:气泡 $=\frac{7}{15}\approx47\%$(近一半空转,太亏);$p=8,m=64$:气泡 $=\frac{7}{71}\approx9.9\%$;$m=128$:$\approx5.2\%$。**结论:micro-batch 数 $m$ 要远大于 stage 数 $p$**(经验 $m\ge4p$)才能把气泡压到可接受。这也限制了 PP 度数不能太大(否则要么气泡大、要么 micro-batch 多到显存吃紧),配 1F1B 调度进一步压气泡(见 [[072 流水线并行与气泡|流水线气泡]])。

**7. 每卡显存:并行怎么省(代入 70B)。** 70B、混合精度约 16 字节/参数,模型态(权重+梯度+优化器)$\approx 1120$GB:

- **纯 DP**:每卡全副本 1120GB → 单卡 80GB **放不下**(DP 不省单卡模型显存)。
- **ZeRO-3 / FSDP**(切优化器+梯度+参数到 64 卡):每卡 $\approx1120/64\approx17.5$GB,**一步到位**。
- **若单层都超卡**:再叠 TP=8(机内),把单层权重和激活也切 8 份。
- **激活**另算(见 [[065 梯度检查点与激活重计算|重计算]]),长序列下是另一大头。

可见**选并行的第一性问题是「每卡显存账」**:先用 ZeRO/FSDP 切冗余,不够再上 TP/PP。

**8. ZeRO 三阶段与通信代价。** ZeRO-1 只切优化器状态(省最多、几乎不加通信);ZeRO-2 再切梯度;ZeRO-3 / FSDP 连参数也切——前向/反向用到某层参数时临时 all-gather、用完即弃,**省显存最多但每步多两次 all-gather + 一次 reduce-scatter**,通信量约为纯 DP 的 1.5 倍。规律:**切得越多越省显存、通信越重**,按显存紧张程度逐级上(1→2→3)。

## 代码

```python
# 伪代码：3D 并行的设备网格（device mesh）划分 —— 把 N 卡排成 (DP, PP, TP) 立方
def build_mesh(world_size, dp, pp, tp):
    assert world_size == dp * pp * tp, "三维乘积必须等于总卡数"
    # rank -> (dp_idx, pp_idx, tp_idx)
    mesh = {}
    for r in range(world_size):
        tp_idx = r % tp
        pp_idx = (r // tp) % pp
        dp_idx = r // (tp * pp)
        mesh[r] = (dp_idx, pp_idx, tp_idx)
    return mesh

# 1024 卡 = DP16 × PP8 × TP8；TP 组限在机内（连续 8 卡同机），DP 组跨节点
print(build_mesh(1024, dp=16, pp=8, tp=8)[0])   # rank0 -> (0,0,0)
print(build_mesh(1024, dp=16, pp=8, tp=8)[8])   # rank8 -> (0,1,0)  下一 PP stage

# ❌ 错：把 TP 度数设得跨机（如 TP=16 跨 2 机）
#    每层 all-reduce 走慢速跨机网络 → 通信掐死，吞吐暴跌
# ✅ 对：TP 限机内 NVLink（≤单机卡数），跨机用通信轻的 PP / DP
```

```python
# 选型决策树（伪代码）
def choose_parallel(model_fits_single_card, is_moe, long_ctx, layer_fits, deep):
    plan = []
    if model_fits_single_card:
        return ["DP"]                       # 最简单，优先
    plan.append("ZeRO/FSDP")                # 先切冗余，常够用
    if not layer_fits: plan.append("TP(机内)")
    if deep:           plan.append("PP(跨机)")
    if is_moe:         plan.append("EP")
    if long_ctx:       plan.append("SP/CP")
    return plan
print(choose_parallel(False, True, True, False, True))
# ['ZeRO/FSDP', 'TP(机内)', 'PP(跨机)', 'EP', 'SP/CP']  —— 典型超大 MoE 长上下文配方
```

```python
# —— 流水线气泡占比:micro-batch 数 m 要远大于 stage 数 p ——
def bubble(p, m): return (p - 1) / (m + p - 1)
for p, m in [(8, 8), (8, 32), (8, 64), (8, 128)]:
    print(f"p={p} m={m}: 气泡={bubble(p, m)*100:.1f}%")
# m=p 时 ~47%(惨);m≥4p 才压到 ~10% 以下 → 经验 micro-batch ≥ 4×stage

# —— Ring AllReduce 每卡通信量与 N 几乎无关(DP 可扩展的根因) ——
def ring_allreduce_bytes(M, N): return 2 * M * (N - 1) / N   # ≈2M,与N无关
for N in [8, 64, 512]:
    print(f"N={N:>3} 卡: 每卡收发 ≈ {ring_allreduce_bytes(1.0, N):.3f}·M")  # 都≈2M

# —— 每卡显存:70B 用 ZeRO-3 切到 64 卡 ——
N_param, bytes_per = 70e9, 16          # 混合精度 ~16 字节/参数(权重+梯度+优化器)
total = N_param * bytes_per
for shards in [1, 64]:
    print(f"分片到 {shards:>2} 卡: 每卡模型态 ≈ {total/shards/1e9:.0f}GB")
# 1(纯DP)→1120GB 放不下;64(ZeRO-3/FSDP)→17.5GB 一步到位
# ❌ 纯 DP 不省单卡模型显存;✅ 先 ZeRO/FSDP 切冗余,不够再叠 TP/PP
```

## 面试高频

- **Q:DP、TP、PP 分别切什么?** A:DP 切 batch(样本维),每卡放整个模型副本,梯度 all-reduce 同步;TP 切单层内部的权重矩阵(隐藏维),每层 all-reduce 拼回;PP 切层(深度维),每卡放连续几层,激活点对点传递。三者正交可叠成 3D 并行。
- **Q:为什么 TP 要限在机内,PP 可以跨机?** A:TP 每层都要 all-reduce 同步激活,通信量大、频率高,必须走机内 NVLink 高带宽;PP 只在 stage 边界点对点传一次激活,通信轻,跨机带宽够用。口诀:TP 通信最重、PP 最轻。
- **Q:模型放不下时,先上 TP 还是先上 ZeRO?** A:一般先 ZeRO/FSDP——它本质是改良 DP,切掉优化器态/梯度/参数的冗余,实现简单且常一步到位;ZeRO 还不够(单层都超卡)再加 TP,层太多再加 PP。
- **Q:EP 和 TP 有何不同?** A:EP 切 MoE 专家(按专家编号整块分卡),token all-to-all 送去送回,只在 MoE 层;TP 切单层矩阵(隐藏维),每层 all-reduce。两者正交,大 MoE 常 EP×TP 叠用。详见 [[049 专家并行 EP 与 MoE 部署|EP]]。
- **Q:什么是 3D 并行?** A:同时用 DP×TP×PP 三个维度切,总卡数=三者乘积。典型如 TP 限机内、PP 跨机、DP 再复制多份,Megatron-Turing NLG 530B、各大千亿模型都这么训。
- **Q:为什么 DP 扩展性最好?** A:Ring AllReduce 每卡收发约 $2M$ 字节(与卡数 $N$ 几乎无关,是常数),且能与反向计算重叠几乎隐藏掉;加卡不增加每卡通信量。对比 TP 每层都 all-reduce 激活、通信随频率累积。
- **Q:流水线气泡多大,怎么压?** A:占比 $\frac{p-1}{m+p-1}$;$p=8,m=8$ 高达 47%,$m=64$ 降到 ~10%。经验 micro-batch 数 $m\ge4p$,再配 1F1B 调度。所以 PP 度数不能太大。
- **Q:ZeRO 三阶段区别?** A:ZeRO-1 切优化器态(省最多、几乎不加通信),ZeRO-2 再切梯度,ZeRO-3/FSDP 连参数也切(用时 all-gather、用完弃,省最多但每步多次通信约 1.5× DP)。按显存紧张逐级上。
- **Q:70B 模型每卡显存怎么估、怎么放下?** A:混合精度 ~16 字节/参数,模型态 ~1120GB,纯 DP 每卡全副本放不下;ZeRO-3/FSDP 切到 64 卡 → 每卡 ~17.5GB,常一步到位;单层超卡再叠 TP。
- **陷阱**:① DP 不省单卡模型显存,只解决算得慢;② TP 跨机会被网络掐死;③ PP 有气泡,微批要够多($m\ge4p$);④ 各维度乘积必须整除总卡数;⑤ ZeRO 切得越多省显存越多但通信越重。

## 关键事实

- **数据并行 + 梯度 all-reduce** 是最基础范式,PyTorch DDP 标准实现;通信可与反向重叠,扩展性最好(见 [[069 数据并行与 AllReduce|数据并行]])。
- **张量并行 Megatron**:Shoeybi 等(2019,arXiv:1909.08053)按行/列切 attention 与 MLP,每个 transformer 层前向 2 次、反向 2 次 all-reduce;限机内高带宽。
- **流水线并行 GPipe / PipeDream**:Huang 等(GPipe,2019,arXiv:1811.06965)用微批减气泡,气泡占比 $\frac{p-1}{m+p-1}$;PipeDream(Narayanan 等,2019)提出 1F1B 调度进一步压气泡。
- **ZeRO**:Rajbhandari 等(2020,SC'20,arXiv:1910.02054)切优化器态/梯度/参数消除 DP 冗余;FSDP 是 PyTorch 原生等价实现。
- **专家并行 EP**:GShard(2020,arXiv:2006.16668)、Switch Transformer(2021,arXiv:2101.03961)系统化,每 MoE 层 2 次 all-to-all。
- **3D 并行落地**:Megatron-Turing NLG 530B(Smith 等,2022,arXiv:2201.11990)用 DP×TP×PP 在数千 GPU 上训练,是经典工程范本。
- 关联:各维度展开 [[069 数据并行与 AllReduce|DP]]、[[070 ZeRO 与 FSDP|ZeRO/FSDP]]、[[071 张量并行(Megatron)|TP]]、[[072 流水线并行与气泡|PP]]、[[049 专家并行 EP 与 MoE 部署|EP]]、[[073 序列并行与上下文并行|SP]];底层 [[074 通信原语与计算通信重叠|通信原语]];显存账 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]]。
