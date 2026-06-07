[[005 多头注意力 Multi-Head|多头注意力]] —— 把 $d$ 维表示切成 $h$ 份(每份 $d/h$ 维),**并行**做 $h$ 个独立的 [[004 缩放点积注意力(为何除以根号 dk)|缩放点积注意力]],各头在不同子空间里学不同关系(语法、指代、远距搭配……),最后把 $h$ 个输出拼接再过一个线性层 $W^O$ 融合。算力与单头基本持平,表达力却大增。

## 直觉
一个人读句子只盯一个角度容易漏信息。多头 = **请 $h$ 个专家同时读同一句话,各看一个维度**:1 号专家盯"主谓搭配",2 号盯"代词指代谁",3 号盯"修饰关系"……每个专家在自己那块 $d/h$ 维的小空间里独立打分、加权。读完后开个碰头会(concat + $W^O$),把各专家的发现揉成一份综合理解。

关键点:不是把 $d$ 维做 $h$ 遍(那样算力翻 $h$ 倍),而是**把 $d$ 维切成 $h$ 段、每段只 $d/h$ 维**。每个头变"瘦"了,$h$ 个头加起来的算力 ≈ 一个 $d$ 维的头。用同样的钱,买到了"多视角"。

## 例子
取 $d=8$,$h=2$,故每头 $d/h=4$。3 个 token(沿用 002 的 `猫 累 它`)。

**① 切分**:整层投影出 $Q,K,V\in\mathbb{R}^{3\times8}$,把最后一维 8 切成 2 段:
- Head 1 拿 $Q_{[:,0:4]},K_{[:,0:4]},V_{[:,0:4]}$,各 $(3,4)$;
- Head 2 拿 $Q_{[:,4:8]},K_{[:,4:8]},V_{[:,4:8]}$,各 $(3,4)$。

**② 各头独立做注意力**(缩放因子是每头维度 $\sqrt{4}=2$,不是 $\sqrt8$):
- Head 1 可能学到"它→猫"的指代,输出 $z^{(1)}\in\mathbb{R}^{3\times4}$;
- Head 2 可能学到"累→猫"的主谓,输出 $z^{(2)}\in\mathbb{R}^{3\times4}$。

**③ 拼接**:$\text{concat}(z^{(1)},z^{(2)})\in\mathbb{R}^{3\times8}$,恢复成 8 维。

**④ 输出投影**:$\times W^O\in\mathbb{R}^{8\times8}$,把两头信息混合成最终 $(3,8)$。$W^O$ 让"各头发现"互相交流——没有它,各头输出只是简单拼一起、互不通气。

数值直觉:第 3 个 token "它" 的最终向量 = [Head1 给的指代信息 ‖ Head2 给的搭配信息],再经 $W^O$ 线性融合。

**拼接 + $W^O$ 数字走一遍**。设 Head 1 对 "它" 输出 $z^{(1)}_3=[1.0,0.5,0.0,2.0]$,Head 2 输出 $z^{(2)}_3=[0.3,0.0,1.0,0.0]$。拼接:
$$\text{concat}=[\underbrace{1.0,0.5,0.0,2.0}_{\text{head 1}},\ \underbrace{0.3,0.0,1.0,0.0}_{\text{head 2}}]\in\mathbb{R}^8$$
再乘 $W^O\in\mathbb{R}^{8\times8}$。$W^O$ 的每一行从**两个头各取一部分**做线性组合——比如某个输出维 $=0.5\cdot(\text{head1 第1维})+0.2\cdot(\text{head2 第3维})+\dots$,这就让"指代信息"和"搭配信息"在输出里**交叉混合**。没有 $W^O$,输出就只是两段信息硬拼、井水不犯河水,头与头之间永远不通气。

**参数量精算(高频追问)**。多头不增加参数:$h$ 个头各有 $W_i^Q,W_i^K\in\mathbb{R}^{d\times(d/h)}$、$W_i^V\in\mathbb{R}^{d\times(d/h)}$,把所有头的 $W_i^Q$ 横向拼起来正好是一个 $d\times d$ 的大矩阵。所以:
$$\text{QKV 投影} = 3\times d\times d = 3d^2,\quad W^O = d^2,\quad \text{合计}\ 4d^2$$
和"单个 $d$ 维头 + 输出投影"参数量**完全一样**($4d^2$)。多头是把同样的 $4d^2$ 参数**重新切分**成多个子空间,不是叠加。FLOPs 同理守恒。

![[tf-多头切分.png]]

## 原理
每个头 $i$ 用**自己的一套**投影:
$$\text{head}_i=\text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V),\quad W_i^Q,W_i^K\in\mathbb{R}^{d\times d_k},\ W_i^V\in\mathbb{R}^{d\times d_v}$$
其中 $d_k=d_v=d/h$。$h$ 个头拼接后线性融合:
$$\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O,\quad W^O\in\mathbb{R}^{h d_v\times d}=\mathbb{R}^{d\times d}$$

**张量形状(实现视角,带 batch B、序列 L)**:
1. 输入 $X:(B,L,d)$,投影出 $Q,K,V:(B,L,d)$;
2. reshape + transpose 成 $(B,h,L,d/h)$ —— 把"头"提到独立维度,方便并行;
3. 打分 $QK^\top:(B,h,L,L)$,缩放 $\sqrt{d/h}$,行 softmax;
4. $\times V:(B,h,L,d/h)$;
5. transpose 回 + reshape 成 $(B,L,d)$(即 concat);
6. $\times W^O:(B,L,d)$。

**为什么多头更强(直觉 + 论文论据)**:
- **多子空间**:单个 softmax 注意力倾向于"聚焦一处";多头让模型同时关注**多个不同位置/不同表示子空间**的信息(论文原话:"jointly attend to information from different representation subspaces at different positions")。
- **平均的副作用被规避**:单头里 softmax 加权求和会"平均掉"细节;切成多头后,每个头可保留各自的尖锐关注,最后再融合。
- **算力守恒**:每头维度降到 $d/h$,总 FLOPs 与单个 $d$ 维头相当;$\sqrt{d/h}$ 缩放也随之改变。

注意:多头不是"集成 $h$ 个完整注意力",而是"在切分后的低维子空间各做一个"。整体放进 block 的数据流见 013,矩阵形式见 006。

**头数 $h$ 怎么选?** 经验上 $d/h$(每头维度)取 64 左右最常见:$d=512,h=8$;$d=768,h=12$;$d=4096,h=32$,都让 $d/h=64$。头太少(每头维度大)失去多视角;头太多(每头维度太小,如 $d/h<16$)单头表达力不足、且小矩阵乘对硬件不友好。$h$ 必须整除 $d$。

**头会冗余吗?——剪枝研究**。Michel et al.(2019, *Are Sixteen Heads Really Better than One?*)发现:训练好的多头模型里,**很多头可以剪掉而几乎不掉点**,说明头之间有冗余、且重要性不均。但训练阶段多头仍有用——它提供了优化时的多样性和容错。这是"多头到底多重要"的经典追问。

**推理省显存:MQA 与 GQA(现代 LLM 必考)**。多头推理时要为每个头缓存 K、V(见 [[102 KV-Cache|KV-Cache]]),$h$ 个头的 KV-Cache 是显存大头。两种省法:
- **MQA(Multi-Query Attention,Shazeer 2019)**:所有 query 头**共享同一组 K、V**(只 1 套 KV)。KV-Cache 缩小 $h$ 倍,推理快很多,但质量略降。
- **GQA(Grouped-Query Attention,Ainslie et al. 2023)**:折中——把 $h$ 个 query 头分成 $g$ 组,每组共享一套 K、V($g$ 套 KV,$1<g<h$)。LLaMA-2/3、Mistral 等普遍采用,在质量和显存间取平衡。

记法:MHA 是 $h$ 套 KV,GQA 是 $g$ 套($g$ 组),MQA 是 1 套。query 头数始终是 $h$,变的只是 KV 头数。

## 代码
```python
import numpy as np

def softmax(x):
    x = x - x.max(-1, keepdims=True); e = np.exp(x)
    return e / e.sum(-1, keepdims=True)

def multi_head_attention(X, Wq, Wk, Wv, Wo, h):
    B, L, d = X.shape
    dh = d // h
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                 # (B,L,d)
    def split(t):                                     # (B,L,d) -> (B,h,L,dh)
        return t.reshape(B, L, h, dh).transpose(0, 2, 1, 3)
    Q, K, V = split(Q), split(K), split(V)
    scores = Q @ K.transpose(0,1,3,2) / np.sqrt(dh)   # (B,h,L,L) 缩放用 dh!
    A = softmax(scores)
    ctx = A @ V                                       # (B,h,L,dh)
    ctx = ctx.transpose(0,2,1,3).reshape(B, L, d)     # concat -> (B,L,d)
    return ctx @ Wo                                   # (B,L,d)

B, L, d, h = 2, 3, 8, 2
X = np.random.randn(B, L, d)
Wq=Wk=Wv=Wo=np.random.randn(d, d)
print(multi_head_attention(X, Wq, Wk, Wv, Wo, h).shape)  # (2, 3, 8)

# ❌ 朴素错误 1:缩放用 √d 而非 √(d/h) → 每头维度其实是 d/h,量级算错
# ❌ 朴素错误 2:忘了最后的 Wo → 各头输出只是拼一起,从不互相融合
# ❌ 朴素错误 3:reshape 后不 transpose → 把"头"和"序列"维搅在一起,语义全错
```

## 面试高频
- **Q:多头注意力比单头好在哪?** A:在多个表示子空间、多个位置上并行关注不同关系,避免单头把信息平均掉;且每头维度 $d/h$,总算力与单头相当。
- **Q:8 个头是不是算力 ×8?** A:不是。把 $d$ 切成 8 份,每头维度 $d/8$,总 FLOPs ≈ 单个 $d$ 维头。多的是表达力,不是算力。
- **Q:多头里缩放因子是多少?** A:$\sqrt{d/h}$(每头维度),不是 $\sqrt{d}$。
- **Q:$W^O$ 有什么用?** A:把各头拼接后的输出做一次线性融合,让头间信息交流并投回 $d$ 维;去掉它各头互不通气。
- **Q:头数 $h$ 怎么定?** A:经验上让每头维度 $d/h\approx64$;$h$ 须整除 $d$;太少失多视角,太多每头表达力不足且小矩阵乘低效。
- **Q:多头是不是冗余的?** A:有一定冗余——Michel et al.(2019)发现训练后很多头可剪掉而不掉点;但训练阶段多头提供优化多样性,仍有价值。
- **Q:MQA、GQA、MHA 区别?为什么需要?** A:MHA 每个 query 头各有 KV($h$ 套);MQA 所有 query 头共享 1 套 KV;GQA 折中分 $g$ 组共 $g$ 套 KV。目的是缩小推理时的 KV-Cache、加速长上下文推理(MQA 缩 $h$ 倍,GQA 缩 $h/g$ 倍),代价是质量略降。LLaMA-2/3、Mistral 用 GQA。
- **Q:为什么多头比"把单头做 $h$ 遍再平均"好?** A:多头是在**不同子空间**各做一个(每头维 $d/h$),能捕获不同类型关系并保留各自尖锐关注;"单头做 $h$ 遍"是同一子空间重复,既贵又无新信息。
- **陷阱**:实现上务必 reshape 成 $(B,h,L,d/h)$ 并把 $h$ 提为独立维度;别忘 $W^O$;缩放用每头维度 $\sqrt{d/h}$ 不是 $\sqrt{d}$;别说"多头算力 ×$h$"(参数和 FLOPs 守恒 $4d^2$);别把 MQA/GQA 的"共享"理解成"query 头也共享"——共享的只有 K、V,query 头始终独立。延伸:推理时多头的 K、V 要缓存(见 [[102 KV-Cache|KV-Cache]]),GQA/MQA 通过让多个 query 头共享 K/V 头来省显存。

## 关键事实
- 多头注意力出自 Vaswani et al. 2017(arXiv:1706.03762)第 3.2.2 节;原文 $h=8$,$d_{model}=512$,$d_k=d_v=d_{model}/h=64$。
- 论文动机原话:多头允许模型 "jointly attend to information from different representation subspaces at different positions",单头会因平均而抑制这一能力。
- 标准张量形状约定:$(B,H,L,d/H)$,打分矩阵 $(B,H,L,L)$ —— 这是几乎所有实现(PyTorch `nn.MultiheadAttention`、HuggingFace)的内部布局。
- 后续变体:MQA(Shazeer 2019)、GQA(Ainslie et al. 2023, arXiv:2305.13245)在多 query 头间共享 K/V 头,主要为推理省显存。
