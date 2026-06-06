[[049 专家并行 EP 与 MoE 部署|专家并行 EP 与 MoE 部署]] 讲的是：MoE 模型专家太多、一张卡装不下,于是把不同专家**分散到不同 GPU**(专家并行 Expert Parallelism, EP),token 在每个 MoE 层经**两次 all-to-all 通信**飞到目标专家所在的卡上算、再飞回来。它是把 [[042 MoE 动机：稀疏激活与容量解耦|稀疏 MoE]] 真正部署到多卡的关键工程手段。

## 直觉

MoE 的卖点是**总参数巨大、单 token 只激活一小部分**(见 [[042 MoE 动机：稀疏激活与容量解耦|稀疏激活]]):比如 8 个专家、每个 FFN 都是几十亿参数,加起来上百亿,但每个 token 只过 top-2 个。问题来了——**这么多专家的权重,一张 GPU 的显存根本放不下**。

最自然的切法:**一张卡放几个专家,谁也别全装**。GPU0 放专家 0、1,GPU1 放专家 2、3……这就是**专家并行 EP**。它和[[068 并行总览：DP、TP、PP、EP、SP|并行总览]]里别的并行维度的区别在于:切的不是「某一层的矩阵」(那是 TP),也不是「不同的层」(那是 PP),而是**按专家编号切**。

但麻烦也随之而来:**token 在 GPU0,可它的 top-2 专家偏偏在 GPU1 和 GPU3**。于是必须把 token「寄」过去:

1. **all-to-all dispatch(派发)**:每张卡根据 router 的选择,把本卡 token 发往**它该去的专家所在的卡**。这是个全交换——每张卡都既往外发、又往里收。
2. 各卡用**本地专家**算 FFN。
3. **all-to-all combine(回收)**:把算完的结果**飞回 token 原来的卡**,按门控权重加权,接回主干。

一句话:**EP = 专家摊到多卡 + 每个 MoE 层两次 all-to-all 把 token 送去送回**。这两次通信常占整层一半时间,是 MoE 部署的头号瓶颈。

## 例子

**8 token、4 卡、4 专家、top-1 路由**,看一次 dispatch 怎么走。

| token | 所在卡 | router 选中专家 | 专家在哪张卡 | 要搬去 |
|---|---|---|---|---|
| t0 | GPU0 | E2 | GPU2 | → GPU2 |
| t1 | GPU0 | E0 | GPU0 | 留本地 |
| t2 | GPU1 | E0 | GPU0 | → GPU0 |
| t3 | GPU1 | E3 | GPU3 | → GPU3 |
| t4 | GPU2 | E2 | GPU2 | 留本地 |
| t5 | GPU2 | E1 | GPU1 | → GPU1 |
| t6 | GPU3 | E3 | GPU3 | 留本地 |
| t7 | GPU3 | E0 | GPU0 | → GPU0 |

dispatch 后 GPU0 上聚了 {t1, t2, t7} 三个 token 都要过 E0;GPU3 上聚了 {t3, t6} 过 E3。**注意 GPU0 收了 3 个、GPU2 只收 2 个——负载不均**。这就是为什么要[[048 路由稳定性：router z-loss|负载均衡损失]]和 **capacity factor**(每个专家设容量上限,超了的 token 被丢弃/走残差),否则慢卡拖死整批(木桶效应)。算完再 combine 把每个结果送回原卡原位置。

**通信量估算**。隐藏维 `h=4096`、每张卡 `T` 个 token、`E` 张卡。dispatch 时每张卡平均往外发 `T·(E-1)/E` 个 token、每个 `h` 维 fp16(2 字节)。`T=8192`、`E=8` 时,单卡单次 dispatch 约 $8192\times\frac{7}{8}\times 4096\times 2\approx 117$ MB,combine 再来一遍。每层两次、几十层叠起来,通信轻松成主导项——所以工程上拼命做[[074 通信原语与计算通信重叠|计算通信重叠]]。

**EP vs TP 的显存账(为什么 MoE 偏爱 EP)**。同样把一层 MoE 摊到 4 卡:

- **TP 切矩阵**:每个专家的权重都被横/纵切成 4 份,每卡存所有专家的 1/4;每层要 all-reduce 拼回部分和,且每卡都参与每个专家的计算。
- **EP 切专家**:每卡完整存 $E/4$ 个专家;只有被路由到本卡专家的 token 才在本卡算,通信是 all-to-all(只搬 token,不搬权重)。

MoE 专家数多、彼此独立,**按专家整块切(EP)比按矩阵切(TP)通信更省、负载更自然**——这就是 MoE 部署默认上 EP 的原因。两者还能叠:EP 切专家、TP 再切单个专家内部的大矩阵(`EP×TP`)。

![[moe-专家并行-all2all.svg]]

## 原理

**1. MoE 层的计算。** 设输入 token 表示 $x\in\mathbb{R}^{h}$,router 是个线性层 $W_r\in\mathbb{R}^{h\times E}$($E$ 个专家):

$$g=\mathrm{softmax}(W_r^\top x)\in\mathbb{R}^{E},\qquad \mathcal{T}=\text{top-}k(g)$$

输出是被选中专家的加权和:

$$y=\sum_{e\in\mathcal{T}} g_e\,\mathrm{FFN}_e(x)$$

稠密 FFN 每个 token 都过同一个;MoE 里 $\mathrm{FFN}_e$ 的权重**散在不同卡**,所以 $x$ 必须先被搬到第 $e$ 个专家那张卡。

**2. 两次 all-to-all。** 设全局有 $N$ 个 token、$E$ 张卡。dispatch 把张量从「按 token 所在卡分布」**重排**成「按目标专家分布」:每张卡 $i$ 给卡 $j$ 发「在卡 $i$ 上、但要去卡 $j$ 专家」的那些 token。这正是 all-to-all 原语(见 [[074 通信原语与计算通信重叠|通信原语]])。专家算完后 combine 是它的逆操作,把结果送回原位:

$$\underbrace{x_{\text{(按卡)}}}_{\text{dispatch}}\xrightarrow{\text{all-to-all}} x_{\text{(按专家)}}\xrightarrow{\mathrm{FFN}_e}\ y_{\text{(按专家)}}\xrightarrow[\text{combine}]{\text{all-to-all}}\ y_{\text{(按卡)}}$$

**3. capacity factor 与丢弃。** 给每个专家设容量 $C=\lceil \text{cf}\cdot \frac{kN}{E}\rceil$($\text{cf}$ 常取 1.0~1.25)。超过 $C$ 的 token 被丢弃(只走残差,不过专家)。这把不规则的动态路由变成**固定形状的张量**,让 all-to-all 能用规整通信、不爆显存——代价是丢 token(轻微掉点)。GShard、Switch Transformer 都靠它。

**4. EP 与别的并行正交。** EP 切专家、TP 切单个专家内部的矩阵、DP 复制整套、PP 切层。大模型常**叠加**:`EP×TP×DP`。EP 的 all-to-all 发生在 MoE 层;非 MoE 部分(attention、稠密 FFN)走普通 TP/DP。完整谱系见 [[068 并行总览：DP、TP、PP、EP、SP|并行总览]]。

![[moe-EP-vs-TP-显存.svg]]

## 代码

```python
import numpy as np

# —— 用 numpy 模拟一次 all-to-all dispatch（单机演示 EP 的搬运逻辑）——
np.random.seed(0)
E = 4                      # 4 张卡 / 4 个专家（top-1）
T_per = 2                  # 每张卡 2 个 token
h = 8                      # 隐藏维

# 每张卡的 token 表示 + router 选中的专家编号
tokens = {g: np.random.randn(T_per, h) for g in range(E)}
choice = {0: [2, 0], 1: [0, 3], 2: [2, 1], 3: [3, 0]}   # 见例子表

# ✅ dispatch：按「目标专家」把 token 分桶 —— 这就是 all-to-all 干的事
buckets = {e: [] for e in range(E)}                      # 每个专家收到的 token
origin  = {e: [] for e in range(E)}                      # 记下来源以便 combine 送回
for g in range(E):                                       # 遍历每张卡
    for idx, e in enumerate(choice[g]):                  # 遍历卡上每个 token
        buckets[e].append(tokens[g][idx])
        origin[e].append((g, idx))
for e in range(E):
    n = len(buckets[e])
    print(f"专家 E{e} 收到 {n} 个 token  来源={origin[e]}")
# 专家 E0 收到 3 个 -> 负载不均！需要 capacity factor / 负载均衡损失

# combine 就是按 origin 把专家输出写回原卡原位置（此处略）

# ❌ 错：以为 MoE 推理 = 跑全部专家（那是 8 倍算力，白瞎了稀疏性）
#   for e in range(E): y += FFN[e](x)        # 全跑，等于稠密，没省算力
# ✅ 对：只跑 router 选中的 top-k，且 token 要先 all-to-all 到专家所在卡
```

```python
# capacity factor：把动态路由裁成固定形状，all-to-all 才好做
def apply_capacity(counts, cf=1.0, k=1, N=8, E=4):
    C = int(np.ceil(cf * k * N / E))          # 每专家容量上限
    dropped = {e: max(0, c - C) for e, c in counts.items()}
    return C, dropped
print(apply_capacity({0:3, 1:1, 2:2, 3:2}))   # (2, {0:1,...}) -> E0 超 1 个被丢
```

## 训练 vs 推理:部署的两套打法

EP 的痛点在训练和推理下表现不同,面试爱区分:

- **训练**:batch 大(每卡几千 token),专家利用率高、all-to-all 摊得开;主攻**计算-通信重叠**(算上一层 FFN 时预取下一层的 dispatch)、设备受限路由(限制每 token 发往的节点数)。DeepSeek-V3 用 DualPipe 等手段把通信几乎藏进计算。
- **推理**:batch 小(尤其单用户逐 token 解码),每步只有少数 token,**专家利用率极低**、all-to-all 占比飙升 → EP 常不划算。对策:① **专家全复制**(每卡都存全部专家,无需 all-to-all,但显存爆);② **专家卸载/offload**(把冷门专家放 CPU/磁盘,用时换入);③ **批量聚合**多个请求提高利用率;④ 蒸馏成稠密小模型部署。

一句话:**训练靠重叠把通信藏起来,推理靠复制/卸载/聚合绕开 all-to-all**。

## 显存到底要多少:MoE 部署的硬约束

MoE 推理显存几乎全花在「常驻的全部专家权重」上(算力虽省、显存不省):

- Mixtral 8×7B 约 47B 参数,bf16 下权重 ≈ $47\times2=94$ GB → 单张 80GB 卡装不下,必须多卡(EP)或量化。
- 4-bit 量化后 ≈ 24GB,可塞进单张 24~48GB 卡——这也是 MoE 量化部署很热的原因。
- DeepSeek-V3 671B,bf16 ≈ 1.3TB,只能多机多卡 + EP。

所以「MoE 省算力」对**延迟/吞吐**友好,但对**显存预算**反而苛刻;选型时这两本账要一起算(见 [[042 MoE 动机：稀疏激活与容量解耦|容量解耦]])。

## 面试高频

- **Q:什么是专家并行 EP?和 TP 有何不同?** A:EP 把不同**专家整块**放到不同 GPU,token 按路由 all-to-all 搬到专家所在卡;TP 把**同一层的矩阵**横/纵切到多卡、每层 all-reduce 拼回。EP 切的是「专家编号」,TP 切的是「权重维度」,两者正交可叠加。
- **Q:MoE 部署的主要通信开销在哪?** A:每个 MoE 层**两次 all-to-all**——dispatch 把 token 派发到目标专家、combine 把结果收回。在多卡/多机下可占整层约 50% 时间,是头号瓶颈;靠计算通信重叠、分组通信缓解。
- **Q:为什么需要 capacity factor?token 会被丢吗?** A:动态路由导致每个专家收到的 token 数不定,而 all-to-all 需要规整张量。capacity factor 给每个专家设容量上限 $C=\text{cf}\cdot kN/E$,超出的 token 被丢弃(只走残差)。会轻微掉点,但换来固定形状、可控显存。
- **Q:专家负载不均怎么办?** A:① 训练加负载均衡辅助损失(见 [[048 路由稳定性：router z-loss|路由稳定性]])让 router 别总挑同几个专家;② capacity factor 限流;③ 推理时可做专家放置/复制热门专家。
- **Q:为什么 MoE 推理显存大但算力省?** A:**全部专家权重都要常驻显存**(总参数巨大),但每 token 只激活 top-k,**FLOPs 只随激活专家走**。所以 MoE 是「用显存换算力」:容量与计算解耦。
- **Q:EP 在推理时为什么常不划算?** 推理 batch 小(逐 token 解码),专家利用率低、all-to-all 占比飙升;对策是专家全复制、专家卸载、批量聚合或蒸馏成稠密。
- **Q:Mixtral 推理要多少显存?** bf16 下 47B×2≈94GB(单 80GB 卡装不下,需多卡或量化);4-bit 量化后约 24GB 可单卡。算力省、显存不省是核心矛盾。
- **Q:EP 和 TP 切 MoE,谁更省?** EP(按专家整块切)对 MoE 更自然:只搬 token 不搬权重、负载更匀;TP 切矩阵每层要 all-reduce 且每卡参与每个专家。两者可叠成 EP×TP。
- **Q:怎么把 all-to-all 通信藏起来?** 计算-通信重叠(算当前层 FFN 时预取下一层 dispatch)、设备受限路由(限每 token 节点数)、DualPipe 等流水排布。
- **陷阱**:① all-to-all 是阻塞型集合通信,慢卡(收 token 最多的)拖垮整批;② EP 度数最好整除专家数;③ 推理 batch 小时通信占比更高,EP 不一定划算,可改 expert 全复制。

## 关键事实

- 专家并行与 all-to-all 通信范式由 **GShard**(Lepikhin 等,2020,arXiv:2006.16668)系统化:自动把专家分片到多设备,引入 all-to-all dispatch/combine 与 capacity factor。
- **Switch Transformer**(Fedus 等,2021,arXiv:2101.03961)用 top-1 路由 + EP 把 MoE 推到万亿参数,在数百 TPU 上高效训练,把 all-to-all 带入大模型主流;并提出容量因子与负载均衡损失的工程范式。
- all-to-all 在多卡/多机分布式下可占 MoE 训练总时间约 50%,是核心瓶颈(多篇 MoE 系统论文测得,如 Shortcut-connected EP,arXiv:2404.05019)。
- 代表落地:GLaM、Mixtral 8x7B、DeepSeek-MoE、Qwen-MoE 等均靠 EP 部署;DeepSeek-V3 进一步做细粒度专家 + 设备受限路由(每 token ≤4 节点)+ DualPipe 计算通信重叠降低跨机 all-to-all。
- 推理对策:专家全复制(免 all-to-all 但显存爆)、专家卸载(冷门专家放 CPU/磁盘)、批量聚合提利用率、蒸馏回稠密;EP 在小 batch 推理常不划算。
- 显存账:Mixtral 47B bf16≈94GB(需多卡/量化,4-bit≈24GB 可单卡);DeepSeek-V3 671B bf16≈1.3TB(多机)。MoE 省算力不省显存。
- EP vs TP:MoE 偏 EP(按专家切、只搬 token、负载自然),可与 TP 叠成 EP×TP。
- 关联:稀疏激活动机 [[042 MoE 动机：稀疏激活与容量解耦|MoE 动机]];路由与负载均衡 [[048 路由稳定性：router z-loss|路由稳定性]];并行全谱 [[068 并行总览：DP、TP、PP、EP、SP|并行总览]];底层通信 [[074 通信原语与计算通信重叠|通信原语与重叠]]。
