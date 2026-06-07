[[069 数据并行与 AllReduce|数据并行与 AllReduce]] 是最基础的分布式训练范式:**每张卡放一份完整模型**,把全局 batch 切成子批分给各卡各自算梯度,再用 **all-reduce** 把各卡梯度求平均同步,然后每卡用同一个平均梯度更新——效果等价于在大 batch 上做一次 [[37 梯度下降：BGD、SGD、Mini-batch|SGD]]。

## 直觉

最朴素的"多卡加速"想法:模型一份份复制到每张卡,把一个大 batch 拆开,每卡只算自己那一小份样本。这就是**数据并行(Data Parallelism, DP)**。它解决的是"**算得太慢**",不是"放不下"——因为每卡都得装下整个模型。

但有个问题:每卡只看到 1/N 的数据,算出的梯度只是局部的。想让训练等价于在整个大 batch 上更新,就必须**把各卡的梯度合并**。具体地,mini-batch SGD 的梯度是样本梯度的平均;把样本分到 4 张卡,全局平均梯度 = 4 张卡本地梯度的平均:
$$\bar g=\frac{1}{4}(g_0+g_1+g_2+g_3).$$
"把各卡的 $g_i$ 求和并把结果发回每张卡",正是 **all-reduce** 这个集合通信原语干的事。每卡拿到同一个 $\bar g$,用同一学率更新 $\theta\leftarrow\theta-\eta\bar g$。因为初始 $\theta$ 相同、梯度也相同,**更新后各卡参数自动保持一致**,无需再同步参数。

一句话:**DP = 复制模型 + 切 batch + 每步梯度 all-reduce 同步**。它是 PyTorch DDP、以及 [[070 ZeRO 与 FSDP|ZeRO/FSDP]] 的基础。

![[dist-数据并行流程.png]]

## 例子

**4 卡、全局 batch B=256**:

1. 切成 4 个子批,每卡 64 样本。每卡用**自己那份完整模型**做前向+反向,得本地梯度 $g_0,\dots,g_3$。
2. **all-reduce**:把 4 个梯度求和并广播回每卡,各卡都得到 $\bar g=(g_0+g_1+g_2+g_3)/4$。
3. 每卡 $\theta\leftarrow\theta-\eta\bar g$。各卡参数仍完全一致。

这等价于单卡上 B=256 的一次更新——所以 DP 是在**放大有效 batch**。这就引出 [[063 批大小、梯度累积与 critical batch size|批大小与梯度累积]]:卡再多,有效 batch 超过 critical batch size 后,收益会递减。

**通信量算一笔(为什么用环形 all-reduce)**。模型 1B 参数、梯度 fp16(2 字节)=2GB。
- **朴素(参数服务器)**:所有卡把 2GB 发给一个 master,master 收 $(p-1)\times2$GB、再广播——master 成单点瓶颈,卡越多越堵。
- **环形 all-reduce**:每卡只与右邻收发,单卡总传输量 $\approx 2N\cdot\frac{p-1}{p}$。$p$ 大时趋于常数 $2N=4$GB,**与卡数几乎无关**——这是 NCCL 默认算法。

![[dist-环形allreduce.png]]

## 原理

**1. 梯度平均的正确性。** mini-batch SGD 在 batch $\mathcal B$ 上的梯度
$$g(\mathcal B)=\frac{1}{|\mathcal B|}\sum_{x\in\mathcal B}\nabla_\theta \mathcal L(x;\theta).$$
把 $\mathcal B$ 划成 $p$ 个不相交子批 $\mathcal B_0,\dots,\mathcal B_{p-1}$(等大),则
$$g(\mathcal B)=\frac{1}{p}\sum_{i=0}^{p-1} g(\mathcal B_i).$$
所以**各卡本地梯度求平均 = 全局大 batch 梯度**,DP 在数学上严格等价于大 batch SGD(子批不等大时改用加权平均)。

**2. all-reduce 原语。** 输入:每卡一个张量 $g_i$;输出:每卡都拿到 $\bigoplus_i g_i$(此处 $\bigoplus$ 取求和,随后乘 $1/p$ 得平均)。它 = **reduce-scatter(分块累加)+ all-gather(广播回去)**,见 [[074 通信原语与计算通信重叠|通信原语]]。

**3. 环形 all-reduce 的带宽最优性。** 把梯度切成 $p$ 块,排成环:
- **reduce-scatter($p-1$ 步)**:每步每卡把一块发右邻、收到的与本地对应块相加。$p-1$ 步后,每卡"攒齐"了某一块的全局和。
- **all-gather($p-1$ 步)**:把各卡攒齐的那块沿环再传一圈,$p-1$ 步后每卡集齐全部 $p$ 块。

单卡总传输 $=2N\cdot\frac{p-1}{p}$($N$=梯度字节)。关键在**与 $p$ 几乎无关**——大规模训练扩展性的根基。

**4. 计算通信重叠(性能关键)。** 反向传播是**逐层从后往前**算梯度的;一旦某层梯度算完,就**立刻开始 all-reduce 这一层**,同时继续算前面层——通信藏在计算背后。PyTorch DDP 用"梯度分桶(bucketing)"实现:梯度攒够一桶就异步 all-reduce。这把通信开销大幅隐藏,见 [[074 通信原语与计算通信重叠|计算通信重叠]]。

**5. DP 的边界。** DP **不省单卡模型显存**(每卡全副本:参数+梯度+优化器态,见 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]])。模型放不下时,要么用 [[070 ZeRO 与 FSDP|ZeRO/FSDP]] 切掉这些冗余,要么上 [[071 张量并行(Megatron)|TP]] / [[072 流水线并行与气泡|PP]]。DP 只解决"算得快"。

## 代码

```python
# PyTorch DDP 标准用法（每卡一个进程，每进程一份完整模型）
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")                 # NCCL 后端走 GPU 高速互联
rank = dist.get_rank()
model = MyModel().cuda(rank)
model = DDP(model, device_ids=[rank])           # 包一层，反向自动 all-reduce 梯度

for batch in loader:                            # DistributedSampler 让各卡拿不同子批
    out = model(batch.cuda(rank))
    loss = out.loss
    loss.backward()                             # ✅ 反向中梯度按桶异步 all-reduce（与计算重叠）
    opt.step(); opt.zero_grad()
# 各卡梯度同步 → θ 始终一致，无需手动同步参数
```

```python
# ❌ 错：手动把所有梯度发到 rank0 累加再广播（参数服务器式，单点瓶颈）
if rank == 0:
    for r in range(1, p): g += recv(r)          # rank0 收 (p-1)·N，卡多就堵死
    for r in range(1, p): send(r, g / p)
# ✅ 对：用 all-reduce（NCCL 默认环形），负载均摊、每卡传输量与 p 无关
import torch.distributed as dist
dist.all_reduce(g, op=dist.ReduceOp.SUM); g /= p   # 一行搞定，带宽最优

# ❌ 错：忘了 DistributedSampler，每卡都喂同一份数据 → 等于没并行（梯度全一样）
# ✅ 对：DistributedSampler(dataset) 按 rank 切分，各卡看不同子批
```

## 环形 all-reduce 通信量逐步手算

很多人背得出"$2N\frac{p-1}{p}$"却讲不清每一步搬多少。这里把它算死。设梯度总字节 $N$、卡数 $p$,把梯度切成 $p$ 等块,每块 $N/p$ 字节。

**阶段一:reduce-scatter($p-1$ 步)。** 把卡排成环 $0\to1\to\cdots\to p{-}1\to0$。第 $k$ 步,每张卡把"某一块"发给右邻、同时从左邻收一块并与本地对应块相加。$p-1$ 步后,每张卡恰好"攒齐"了某一块的全局和(卡 $i$ 攒齐第 $(i+1)\bmod p$ 块)。
- 每步每卡**发** $N/p$ 字节、**收** $N/p$ 字节。
- $p-1$ 步,单卡总发送 $=(p-1)\cdot\frac{N}{p}=\frac{p-1}{p}N$。

**阶段二:all-gather($p-1$ 步)。** 每卡把自己攒齐的那块沿环再传一圈,$p-1$ 步后人人集齐全部 $p$ 块。
- 同样单卡发送 $\frac{p-1}{p}N$。

**合计:单卡总发送 $=2\frac{p-1}{p}N$**(收发对称,收也一样)。代入 $p=8$、$N=2$ GB:$2\times\frac{7}{8}\times2=3.5$ GB;$p=64$:$2\times\frac{63}{64}\times2\approx3.94$ GB;$p\to\infty$:$\to 4$ GB$=2N$。**单卡搬运量被 $2N$ 卡死,与卡数几乎无关**——这就是环形算法"带宽最优"的全部含义。

**对比朴素参数服务器**:master 要收 $(p-1)N$、发 $(p-1)N$,单点流量随 $p$ **线性爆炸**($p=64$ 时 master 单独扛 $126N=252$ GB),且整个集群只用了 master 那一条链路的带宽。环形把流量均摊到 $p$ 条链路,每条链路同时在收发,聚合带宽 $\times p$。

**通信时间模型**(配合 [[074 通信原语与计算通信重叠|通信原语]] 的 $\alpha$-$\beta$ 模型):一次环形 all-reduce 耗时
$$T\approx \underbrace{2(p-1)\alpha}_{\text{延迟项,随 }p\text{ 线性}}+\underbrace{2\frac{p-1}{p}\frac{N}{B}}_{\text{带宽项,}\to 2N/B}.$$
$\alpha$ 是每次发起通信的固定延迟、$B$ 是链路带宽。**小张量延迟项($2(p-1)\alpha$)主导,大张量带宽项主导**——这正是 DDP 要把碎梯度**打包成桶(bucket)**再传的原因:几百个小张量各发一次,延迟项会被 $\alpha$ 拖垮。

## 学率缩放、warmup 与有效 batch

DP 放大的是**有效 batch** $B_{\text{eff}}=p\times b_{\text{local}}$。batch 变大,单步梯度噪声变小、方向更准,但**一个 epoch 的更新步数变少**——要保持收敛速度,得调学率。两条经典规则:

- **线性缩放(Linear Scaling Rule,Goyal 2017)**:batch 放大 $k$ 倍,学率也放大 $k$ 倍。直觉:batch 大 $k$ 倍 → 梯度方差小 $k$ 倍 → 可以放心迈大 $k$ 倍的步。"1 小时训 ImageNet"用 batch 8192、学率 ×256 验证了它。
- **平方根缩放(Krizhevsky 2014)**:学率按 $\sqrt{k}$ 放大,理由是想让 SGD 更新的"噪声尺度"不变。两种规则在不同设定下各有支持,LLM 预训练里更常见的是**先按经验定一个峰值学率,再配 warmup + cosine 衰减**。

**为什么必须 warmup**:大 batch + 大学率在训练**最初几百步**极不稳——此时参数还在随机初值附近,大学率会让 loss 直接 NaN。warmup 让学率从 0 线性爬到峰值(典型 2000 步),把这段危险期"扶"过去。这和 [[063 批大小、梯度累积与 critical batch size|critical batch size]] 是一体两面:超过 critical batch size 后,再加卡(再放大 batch)收益急剧递减——梯度噪声已经够小了,大 batch 提供的额外信息冗余。

## 梯度累积:把"加更多卡"换成"少花钱"

显存不够上更多卡、又想要大有效 batch?用**梯度累积(gradient accumulation)**:在一张(组)卡上连算 $A$ 个 micro-batch,梯度**累加不清零**,攒够 $A$ 次再 all-reduce + step 一次。
$$B_{\text{eff}}=p\times b_{\text{local}}\times A.$$
关键细节(易错点):① loss 要除以 $A$(否则梯度被放大 $A$ 倍,等效学率失控);② DDP 下前 $A-1$ 次反向要用 `model.no_sync()` **跳过 all-reduce**,只在第 $A$ 次才同步——否则每个 micro-batch 都 all-reduce 一次,通信量被白白放大 $A$ 倍,梯度累积的省通信优势全丢。

## DP 数学等价性的边界:哪些操作会破坏等价

"DP = 大 batch SGD"只在**所有运算对 batch 维线性**时严格成立。会破坏等价的典型:
- **BatchNorm**:统计量(均值/方差)在本卡 sub-batch 上算,不是全局——所以严格等价要用 **SyncBN**(跨卡同步统计量)。Transformer 用 LayerNorm(逐 token,不跨 batch),天然没这问题,这也是 LLM 不踩这个坑的原因。
- **梯度裁剪(clip by global norm)**:全局范数应在 all-reduce 后的平均梯度上算;若各卡先各自裁剪再平均,结果 ≠ 单卡大 batch 裁剪。框架默认在同步后裁,正确。
- **不等大子批**(最后一个 batch 凑不齐):严格等价要按样本数**加权平均**,而非简单算术平均;`DistributedSampler` 默认补齐(drop_last 或 pad)来回避。

## 面试高频

- **Q:数据并行的核心流程是什么?** A:每卡放完整模型副本,全局 batch 切成子批各卡算本地梯度,梯度 all-reduce 求平均后每卡用同一平均梯度更新。因初始参数和梯度都相同,更新后各卡参数自动一致,无需同步参数。
- **Q:为什么是梯度求平均,不是求和?** A:mini-batch SGD 的梯度是样本梯度的平均;大 batch 切成 p 个等大子批后,全局梯度=p 个本地梯度的平均。求平均才严格等价于大 batch SGD。all-reduce 先求和再乘 1/p。
- **Q:环形 all-reduce 为什么高效?** A:梯度切成 p 块沿环流动,分 reduce-scatter + all-gather 两阶段,单卡总传输量约 2N·(p−1)/p,p 大时趋于常数 2N、与卡数几乎无关;且无中心节点,负载均摊到每条链路。朴素参数服务器有单点瓶颈。
- **Q:DP 能解决模型放不下的问题吗?** A:不能。DP 每卡都要放完整模型(参数+梯度+优化器态),只解决"算得慢"。放不下要用 ZeRO/FSDP 切冗余,或 TP/PP 切模型本身。
- **Q:DP 里怎么隐藏通信开销?** A:反向逐层算梯度,某层算完就立刻异步 all-reduce 这一层,同时继续算前层(计算通信重叠)。PyTorch DDP 用梯度分桶实现。
- **Q:梯度累积怎么和 DDP 配合,不踩坑?** A:loss 除以累积步数 $A$;前 $A-1$ 次反向用 `no_sync()` 跳过 all-reduce,只在第 $A$ 次同步——否则每个 micro-batch 都通信,放大 $A$ 倍通信量。有效 batch = $p\times b_{\text{local}}\times A$。
- **Q:为什么 LLM 用 DP 不担心 BatchNorm 的等价性问题?** A:Transformer 用 LayerNorm(逐 token、不跨 batch 维),DP 切 batch 不影响它;而 BatchNorm 的统计量跨 batch,需 SyncBN 才严格等价。这是 CV 训练才要操心的坑。
- **Q:环形 all-reduce 单卡到底搬多少字节,怎么算?** A:reduce-scatter 阶段单卡发 $\frac{p-1}{p}N$,all-gather 再发 $\frac{p-1}{p}N$,合计 $2\frac{p-1}{p}N$;$p\to\infty$ 趋于 $2N$、与卡数无关。延迟项 $2(p-1)\alpha$ 随卡数线性增长,故小张量要 bucket。
- **陷阱**:① 忘了 DistributedSampler 各卡喂同样数据等于没并行;② DP 放大有效 batch,超过 critical batch size 收益递减,学率要按 batch 缩放;③ 各卡随机种子/数据增强要保证子批不同;④ 梯度累积忘了 `no_sync()`,通信被放大 $A$ 倍;⑤ warmup 缺失时大 batch + 大学率开局直接 NaN。

## 关键事实

- **PyTorch DDP** 是数据并行标准实现,基于梯度分桶 + 反向中异步 all-reduce 实现计算通信重叠(Li 等,2020,PyTorch Distributed,arXiv:2006.15704)。
- **环形 all-reduce** 由百度(2017,Baidu DeepBench / "Bringing HPC techniques to deep learning")引入深度学习,后被 **Horovod**(Sergeev & Del Balso,2018,arXiv:1802.05799)与 NVIDIA **NCCL** 广泛采用,成为默认算法;单卡通信量与卡数无关。
- 数据并行严格等价于大 batch SGD:全局梯度=各卡本地梯度平均;由此引出大 batch 训练与学率缩放(见 [[063 批大小、梯度累积与 critical batch size|critical batch size]];Goyal 等 2017 "1 小时训 ImageNet" 提出线性缩放学率,arXiv:1706.02677)。
- DP 不省单卡模型显存:每卡存参数+梯度+优化器态(Adam 下约 16 字节/参数,见 [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存估算]]);ZeRO/FSDP 正是为消除此冗余而生(见 [[070 ZeRO 与 FSDP|ZeRO]])。
- 关联:更新规则 [[37 梯度下降：BGD、SGD、Mini-batch|SGD]] 与优化器 [[39 优化器(Momentum、RMSProp、Adam、AdamW)|Adam]];并行全谱 [[068 并行总览：DP、TP、PP、EP、SP|并行总览]];底层原语 [[074 通信原语与计算通信重叠|通信原语与重叠]]。
