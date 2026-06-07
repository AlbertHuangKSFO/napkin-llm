[[023 online softmax 与数值稳定]] 是一种**流式单趟**计算 softmax 的算法:边读数据边维护 running max($m$)与 running sum($\ell$),来一块更新一块,无需先看到全部元素。它既保证了减最大值的数值稳定,又免去把整行(在注意力里就是整个 $n\times n$ 矩阵)存下来——正是 [[024 FlashAttention 1、2：IO 感知精确注意力|FlashAttention]] 能不落 HBM 的数学地基,直接缓解 [[LLM/014 注意力复杂度 O(n²) 与瓶颈|O(n²) 瓶颈]]。

## 直觉类比
朴素 softmax 像考试改卷:必须先翻遍全班卷子找出最高分(全局 max)、再加总(全局 sum)、最后才能算每人占比——要存下整摞卷子。online softmax 像流水线收卷:每收到一摞就更新"目前最高分"和"目前总和";一旦冒出更高分,把之前累计的总和按比例**重新缩放**一下即可,不用回头重看旧卷。

**生活类比(带数字走一遍)**:老师边收卷边算,手里只攥两个数——"目前最高分 m"和"累计总和 ℓ"。第 1 摞卷分数 `[1,3]`:`m=3`,把分数都按"3 分为基准"折算后累加得 `ℓ=1.14`。第 2 摞 `[2,5]` 来了,**冒出个 5 分比 3 高**——基准过时了。老师不必翻回第 1 摞重算,只把旧总和乘个缩放因子 `e^(3−5)=0.135`,就从"按 3 分基准"换算成"按 5 分基准",再把这摞加进去,得 `m=5, ℓ=1.20`。全程只攥 2 个数,从不堆卷子。最后回头让全班一起算,结果也正好是 `ℓ=1.20`——**逐位一致**,所以 online softmax 是精确的,不是近似。这两个数对应注意力里的 running max 与 running sum,正因为不用堆下整张 n×n 名册,显存从 O(n²) 省到 O(n)。

![[flash-023类比改卷running最高分.png]]

## 小数字例子
向量 `[1, 3, 2, 5]`,分成两块 `[1,3]`、`[2,5]`:
- 块1:$m_1=3$,$\ell_1=e^{1-3}+e^{3-3}=0.135+1=1.135$。
- 块2 来了,新 max $m_2=\max(3,5)=5$。先把旧和缩放:$\ell \leftarrow e^{3-5}\cdot1.135=0.135\cdot1.135=0.154$,再加本块:$+e^{2-5}+e^{5-5}=0.050+1$,得 $\ell=1.204$。
- 全局直接算:$\sum e^{x_i-5}=e^{-4}+e^{-2}+e^{-3}+e^{0}=0.018+0.135+0.050+1=1.204$。**完全一致**(精确,非近似)。
- 内存:只存 $(m,\ell)$ 两个标量,而非整行 4 个 exp 值;注意力里就是从 $O(n^2)$ 省到 $O(n)$。

## 原理:running max/sum 递推
朴素稳定 softmax 需三趟(求 max、求和、归一)。online 用 telescoping(伸缩和)把它压成一趟。设已处理部分的状态 $(m^\text{old},\ell^\text{old})$,新来一块 $\{x_k\}$:

$$m^\text{new} = \max\!\big(m^\text{old},\ \max_k x_k\big)$$

$$\ell^\text{new} = e^{\,m^\text{old}-m^\text{new}}\,\ell^\text{old} \;+\; \sum_k e^{\,x_k - m^\text{new}}$$

缩放因子 $e^{m^\text{old}-m^\text{new}}\in(0,1]$ 把"按旧 max 归一的旧和"改写成"按新 max 归一",从而旧数据无需回看。最终 $\text{softmax}(x_i)=e^{x_i-m}/\ell$。在 FlashAttention 里,输出 $O$ 也同步缩放累加。

## 图
![[flash-online-softmax递推.png]]

把上面 `[1,3,2,5]` 的两块递推每一步数字摊开,并与全局直接算对照,验证逐位相等:

![[flash-023online数值手算.png]]

## 代码:三趟 vs 单趟
```python
# ❌ 朴素稳定 softmax:三趟扫描,需先存下整行 x
m = x.max()                     # 趟1
l = (x - m).exp().sum()         # 趟2
out = (x - m).exp() / l         # 趟3  → x 必须整行驻留

# ✅ online softmax:单趟,只维护标量 (m, l)
m, l = -float('inf'), 0.0
for blk in x.split(BLOCK):                 # 流式来块
    m_new = max(m, blk.max().item())
    l = math.exp(m - m_new) * l + (blk - m_new).exp().sum().item()  # 旧和重缩放
    m = m_new
# softmax(x_i) = exp(x_i - m) / l
```

## 面试高频
- **softmax 为何要减最大值?** `exp` 易上溢;减 max 后最大指数为 $e^0=1$,数值稳定,且不改变结果(分子分母同乘常数)。
- **online softmax 是近似吗?** 不是,数学上与全局 softmax 完全等价(精确)。
- **telescoping 缩放因子是什么?** $e^{m^\text{old}-m^\text{new}}$,把旧 running sum 从"旧基准"换算到"新基准",使旧块无需重算。
- **它和 FlashAttention 什么关系?** FlashAttention 用它在分块计算注意力时不必物化 $n\times n$ 矩阵——这是省 HBM 的核心。

## 关键事实
- 出自 Milakov & Gimelshein,"Online normalizer calculation for softmax",arXiv:1805.02867(NVIDIA, 2018),把 softmax 访存从 4 趟降到 3 趟。
- 维护 running max $m$ + running sum $\ell$,靠 telescoping 重缩放旧和,单趟且精确。
- 被 FlashAttention(Dao et al., 2022)采纳,成为分块注意力不落 HBM 的数学基础。
