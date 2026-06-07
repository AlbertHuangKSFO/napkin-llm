[[016 输出层、tied embedding 与 logits|输出层与 tied embedding]] 讲的是 Transformer 的「出口」:最后一层 hidden 向量 `(B, L, d)` 投影到词表大小 `V`,得到 **logits** `(B, L, V)`,再 softmax 成概率分布。**tied embedding(权重绑定)** 让这个输出投影直接复用输入嵌入表的转置 $E^\top$,省下一大块参数还常能提升效果。

## 直觉

网络跑到最后,每个位置手里有一个 `d` 维 hidden 向量,它编码了「这里该接什么词」的信息。但我们要的是**一个词**,而词表有 `V` 个候选(常 3 万~15 万)。于是需要一把「打分尺」:给每个候选词打一个分(logit),分高的更可能是下一个词。

这把尺子怎么打分?**让 hidden 向量和「每个词的向量表示」做点积**——和谁越像,谁的分越高。而「每个词的向量表示」我们其实**已经有了**:就是入口处的嵌入表 $E$(每行一个词的 `d` 维向量,见 [[04 Embedding 与向量数据库|Embedding]])。

既然入口用 $E$ 把「词 → 向量」,出口正好可以用 $E^\top$ 把「向量 → 词的分数」——**同一份词向量,入口编码、出口打分,一鱼两吃**。这就是 **tied embedding**:不再额外学一个输出矩阵,直接绑定 $W_{\text{out}}=E^\top$。好处是省参数(少一份 `V×d`)、且强制输入输出共享同一套词义空间,常让小模型 perplexity 更低。

一句话:**logits = hidden 与每个词向量的点积;tied embedding 让「打分用的词向量」就是「输入查表用的词向量」。**

## 例子

**形状走一遍**。设 `d=768`、词表 `V=50000`,某位置 hidden $h\in\mathbb{R}^{768}$。

- 输出投影:$\text{logits}=h\cdot E^\top$,$E^\top\in\mathbb{R}^{768\times 50000}$ → logits $\in\mathbb{R}^{50000}$,每个词一个分。
- 第 $v$ 个分就是 $\text{logit}_v=h\cdot E_v$($E_v$ 是词 $v$ 的嵌入行)——**hidden 与词 $v$ 向量的点积**。
- softmax:$P(v)=\dfrac{e^{\text{logit}_v}}{\sum_j e^{\text{logit}_j}}$,得 5 万维概率,全正、和为 1。

**省多少参数**。不绑定时需要两份各 `V×d` 的矩阵:输入嵌入 `50000×768` 和输出投影 `50000×768`,共 $2\times 3840\text{万}=7680$ 万参数。绑定后**只一份** `50000×768 ≈ 3840 万`,省掉另一半。在小模型里这块占比惊人:GPT-2 small 总参 124M,而单个 `50257×768` 嵌入就约 3860 万 ≈ **31%**;绑定相当于直接砍掉接近三成参数还不掉点。

**为什么效果还更好**。Press & Wolf(2017)发现:绑定后,梯度从「输出端」也能直接更新词向量,词向量被训得更充分;且共享空间避免了「输入和输出各学一套互不相干的词义」的浪费,在多个语言模型基准上 perplexity 显著下降。

![[tf-tied-embedding.png]]

## 原理

**1. 输出投影 = 与词向量做内积**。设最终(LN 后)hidden 张量 $H\in\mathbb{R}^{B\times L\times d}$,输出权重 $W_{\text{out}}\in\mathbb{R}^{d\times V}$:

$$\text{logits}=H\,W_{\text{out}}\in\mathbb{R}^{B\times L\times V}$$

逐元素看,$\text{logits}_{b,i,v}=\langle H_{b,i},\,(W_{\text{out}})_{:,v}\rangle$:位置 $(b,i)$ 的 hidden 和「词 $v$ 对应的列向量」的内积。通常不加偏置(很多实现 `bias=False`)。

**2. softmax 把分数变概率**(见 [[27 Softmax 与温度|Softmax]]):

$$P(\text{下一词}=v\mid \text{上文})=\frac{\exp(\text{logits}_{b,i,v})}{\sum_{j=1}^{V}\exp(\text{logits}_{b,i,j})}$$

训练时把它和真实下一词的 one-hot 比**交叉熵**(见 [[30 交叉熵与负对数似然|交叉熵]]);推理时按此分布采样或取 argmax。

**3. tied embedding:令 $W_{\text{out}}=E^\top$**。输入嵌入表 $E\in\mathbb{R}^{V\times d}$(查表:$x=E_{\text{id}}$)。绑定即

$$\text{logits}=H\,E^\top,\qquad \text{logit}_{b,i,v}=\langle H_{b,i},\,E_v\rangle$$

**同一个 $E$,前向用两次**:入口按行索引(查表),出口转置后做投影。反向传播时,$E$ 的梯度来自**两条路**——输入侧(被查到的那些行)和输出侧(所有行,因为 softmax 涉及全词表)——所以词向量训得更充分。

**4. 维度匹配的硬约束**。绑定要求输出投影维度 = 嵌入维度 = $d$。若想让嵌入维 $d_{\text{emb}}$ 与模型维 $d_{\text{model}}$ 不同(如 ALBERT 的因子分解嵌入),需在中间插一个 $d_{\text{emb}}\to d_{\text{model}}$ 的小投影,再绑定 $V\times d_{\text{emb}}$ 那一份。

**5. logits 的稳定性**。$V$ 很大时 $\sum e^{z}$ 易上溢,实现里先减最大值:$\text{softmax}(z)_v=\dfrac{e^{z_v-\max z}}{\sum_j e^{z_j-\max z}}$。大模型训练常加 **z-loss**($\lambda(\log\sum e^{z})^2$)防 logits 整体漂移(见 [[015 Transformer 训练稳定性|训练稳定性]])。

**6. 从 logits 到下一个词:解码策略**(推理时怎么用 logits)。logits 经温度 $T$ 调节后采样(见 [[27 Softmax 与温度|温度]]):$P(v)=\text{softmax}(z/T)$。
- $T\to0$:趋近 argmax(贪心,确定但单调);$T=1$:原始分布;$T>1$:更随机、更多样。
- **top-k**:只在 logits 最高的 $k$ 个里采样;**top-p(nucleus)**:只在累积概率达 $p$ 的最小词集里采样。都是为了砍掉长尾噪声词。

所以训练端(交叉熵、用全分布)和推理端(温度 + top-k/p 采样)用的是同一组 logits,只是后处理不同。这把"输出层"和实际生成连了起来。

**7. logit lens(可解释性彩蛋)**。既然 $\text{logits}=H E^\top$,可以把**中间层**的 hidden 也乘 $E^\top$ 提前"翻译"成词分布,看模型在第几层就"想好"了下一个词——这叫 logit lens,是探查 Transformer 内部的经典工具,靠的正是 tied embedding 让中间表示和词表同空间。

![[tf-logits到概率.png]]

## 代码

```python
import torch, torch.nn as nn

V, d = 50000, 768

# ✅ tied embedding:输入嵌入与输出投影共享同一份权重
emb  = nn.Embedding(V, d)              # 入口:token id -> (·, d)
head = nn.Linear(d, V, bias=False)     # 出口:(·, d) -> (·, V)
head.weight = emb.weight               # ← 关键一行:绑定!两者指向同一张量

# 前向
ids = torch.randint(0, V, (2, 4))      # (B=2, L=4)
x = emb(ids)                           # (2, 4, 768)   入口查表
# ... 经过 N 层 Transformer,得到最终 hidden h ...
h = x                                  # 占位:实际是网络输出 (2, 4, 768)
logits = head(h)                       # (2, 4, 50000)  出口投影
print(logits.shape)                    # torch.Size([2, 4, 50000])

# 验证真的共享了内存(改一个,另一个跟着变)
print(emb.weight.data_ptr() == head.weight.data_ptr())   # True

# 参数对比
n_tied   = V * d                       # 38_400_000  绑定:只一份
n_untied = 2 * V * d                   # 76_800_000  不绑:两份
print(f"绑定省下 {n_untied - n_tied:,} 个参数")
```

```python
# ❌ 错:分别建两个独立矩阵 —— 浪费一份 V×d 参数,小模型里可占总量 ~30%
emb  = nn.Embedding(V, d)
head = nn.Linear(d, V, bias=False)     # 独立权重,与 emb 无关
#（也不会从输出端给词向量额外梯度,通常 perplexity 略差）

# ✅ 对:绑定权重(GPT-2 / BERT / T5 默认)
head.weight = emb.weight
```

## 面试高频

- **Q:logit 到底是什么?怎么算出来的?** A:每个候选词的「未归一化分数」。它等于最终 hidden 向量与该词嵌入向量的点积(`logits = H · W_out`),再 softmax 成概率。
- **Q:什么是 tied embedding / 权重绑定?为什么用?** A:让输出投影矩阵 = 输入嵌入表的转置 $E^\top$,共享同一份权重。好处:① 省一份 `V×d` 参数(小模型里可达总量 20–30%);② 输入输出共享词义空间、词向量从两端都得梯度,常降 perplexity(Press & Wolf 2017)。
- **Q:绑定省了多少参数,举个数?** A:`V=50k, d=768` 时,不绑两份 `7680 万`,绑定一份 `3840 万`,省一半;占 GPT-2 small(124M)约 31%。
- **Q:绑定有什么前提/限制?** A:要求输出投影维 = 嵌入维 = $d$。若嵌入维想和模型维不同,得加中间投影(如 ALBERT 因子分解嵌入)再绑定。
- **Q:输出层为什么常不加 bias?** A:softmax 对所有 logit 同加常数不变,bias 的全局分量被吸收;且绑定时 $E^\top$ 本身无 bias,加了反而破坏对称性。多数实现 `bias=False`。
- **Q:推理时怎么从 logits 得到下一个词?** A:温度调节 $\text{softmax}(z/T)$ 后采样;$T\to0$ 趋贪心、$T>1$ 更随机;再用 top-k(取最高 $k$ 个)或 top-p(累积概率 $p$ 的最小集)砍长尾。训练用全分布算交叉熵,推理用采样,共用同一组 logits。
- **Q:tied embedding 时词向量梯度来自几条路?** A:两条——输入侧(被查到的行)和输出侧(softmax 涉及全词表,所有行都更新);所以词向量训得更充分,常降 perplexity。
- **Q:什么是 logit lens?** A:把中间层 hidden 也乘 $E^\top$ 提前翻成词分布,观察模型第几层就"决定"了下一词;依赖 tied embedding 让中间表示与词表同空间。
- **Q:词表很大时输出层为什么贵?有什么省法?** A:输出投影是 $d\times V$,$V$ 达十万级时是显存/算力大头;省法有 adaptive softmax(按词频分层)、采样 softmax(训练时只算部分负样本)、以及把 $V$ 用 BPE/分词控制在合理规模。
- **陷阱**:① 别忘了 softmax 的数值稳定(减 max);② logits 不是概率,别直接当概率用,要 softmax;③ tied 后改其中一个就改了另一个(共享内存),调试时易踩;④ 词表巨大时输出投影是显存/算力大头,有 adaptive softmax / 采样 softmax 等省法;⑤ 嵌入乘 $\sqrt{d}$ 缩放是原版细节,绑定时输入侧缩放、输出侧不缩放,别搞混。

## 关键事实

- 权重绑定出自 Press & Wolf《Using the Output Embedding to Improve Language Models》(EACL 2017,arXiv:1608.05859):绑定后多个语言模型基准 perplexity 显著下降、参数更少;翻译模型可缩小到原来不到一半而不掉点。
- 已成默认实践:GPT-2、BERT、RoBERTa、T5 等普遍绑定输入嵌入与输出投影。
- 原始 Transformer(Vaswani 等,2017,arXiv:1706.03762)即在两处嵌入与 pre-softmax 线性层间共享权重,并把嵌入乘 $\sqrt{d}$ 缩放。
- 输出投影与 softmax 是交叉熵训练目标的最后一环;logit = hidden 与词向量内积,这也是「相似词得相近分」的来源。
- 关联:嵌入与查表 [[04 Embedding 与向量数据库|Embedding]] 与 [[054 词嵌入层与权重绑定|词嵌入层与权重绑定]];softmax/温度 [[27 Softmax 与温度|Softmax]];交叉熵 [[30 交叉熵与负对数似然|交叉熵]] 与 [[32 困惑度 Perplexity|困惑度]];语言模型基础 [[53 语言模型基础(n-gram→神经 LM)|语言模型基础]];整体位置见 [[013 Transformer 整体数据流(逐张量形状)|整体数据流]]。
