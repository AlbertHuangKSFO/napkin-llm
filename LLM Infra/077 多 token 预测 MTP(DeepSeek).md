[[077 多 token 预测 MTP(DeepSeek)|多 token 预测]](Multi-Token Prediction,MTP)最早作为 **训练目标** 出现在 DeepSeek-V3(2024,arXiv:2412.19437,见 [[LLM/039 Mistral、Qwen、DeepSeek 架构选择|DeepSeek 架构]]):在主干之外接若干个轻量 MTP 模块,让模型在训练时**不只预测下一个 token,还预测后面第 2、3… 个 token**。这一目标致密了训练信号、迫使模型规划更远,提升了主模型本身的质量。妙处在于:这些 MTP 模块**推理时可直接当草稿器**做投机解码——因为草稿器与模型同源、词表天然对齐、分布一致,接受率很高。这种"模型自带草稿头"的范式叫 **self-speculation(自投机)**,无需独立草稿模型,也无需像 [[074 EAGLE-3：工业标准投机解码|EAGLE]] 那样额外训练一个草稿头。它是 [[073 投机解码系统：draft-verify 全流程|draft-verify]] 的一种特例,验证仍走无损接受规则,并已在 [[079 投机解码与连续批、前缀缓存的兼容|SGLang/vLLM]] 中作为 DeepSeek-V3 的默认加速路径。

## 直觉

普通模型训练时被要求"只看眼前一步":给定前文,预测**紧接着的那一个字**。这像让学生只会答"下一个字是什么",缺乏对句子走向的规划。

MTP 训练时多压几个担子:同一个隐状态,既要预测 t+1,又要(通过附加的小模块)预测 t+2、t+3。学生被迫**多想几步、规划全局**——结果连"答下一个字"也答得更准,主模型质量提升。

而到了推理,这几个"已经会预测未来 2–3 个字"的模块**正好是现成的实习生**:它们和主编(主模型)同一个师门、用同一套词典、说话风格一致,所以猜得特别准、主编打回得少。别的方法要专门去训练或采购一个实习生;DeepSeek 是"训练主编时顺手把实习生一起带出来了",零额外成本、还同源高接受。

## 例子

DeepSeek-V3:主干 671B(MoE,激活 37B),MTP 模块共 ~14B 参数(每个模块 = 共享 embedding + 共享 output head + 1 个 Transformer 块 + 投影矩阵)。

推理时用 **MTP1**(预测第二个 token,即 t+2)做单步草稿:论文报告 **MTP1 接受率 > 80%**,带来生成吞吐约 **1.8×**。

代入收益模型(草稿长 $k=1$、$\alpha=0.8$):每轮期望接受 $\frac{1-\alpha^{2}}{1-\alpha}=\frac{1-0.64}{0.2}=1.8$ 个 token,与报告的 1.8× 吻合。若用更长草稿(多个 MTP 模块串成 k=2),理论上界更高,但远端 MTP 模块接受率衰减,实践收益边际递减(详见 [[078 接受率、草稿长度与收益分析|收益分析]])。

## 原理

MTP 在主干顶端的 hidden $h_t$ 上,串联 $D$ 个 MTP 模块预测 t+1..t+D。第 $d$ 个模块(DeepSeek-V3 用**因果链式**结构,而非 Medusa 的并行独立头):

$$
h^{(d)}_t \;=\; \mathrm{TRM}_d\!\Big( \mathrm{Proj}_d\big[\,\mathrm{RMSNorm}(h^{(d-1)}_t)\,;\,\mathrm{RMSNorm}(\mathrm{Embed}(x_{t+d}))\,\big]\Big)
$$

$$
p^{(d)}(x_{t+1+d}) \;=\; \mathrm{softmax}\big(\mathrm{Head}_\text{shared}\, h^{(d)}_t\big)
$$

关键:第 $d$ 个模块**条件于前一个模块的输出** $h^{(d-1)}$(链式、保持因果),这比 Medusa 各头独立更连贯,接受率更高。下图核对两者的依赖箭头:MTP 是**横向串行链**(模块逐个传递 $h^{(d-1)}$、保序),Medusa 是从同一 $h_t$ **发散的并行头**(头间无横向箭头、无序列依赖)——这正是 MTP 接受率更高的结构根因:

![[spec-MTP串行vs Medusa发散.png]]

训练损失是各深度 [[深度学习基础/30 交叉熵与负对数似然|交叉熵]] 的加权和:

$$
\mathcal{L}_\text{MTP} \;=\; \lambda \cdot \frac{1}{D}\sum_{d=1}^{D} \mathrm{CE}\big(x_{t+1+d},\, p^{(d)}\big)
$$

推理走 [[073 投机解码系统：draft-verify 全流程|draft-verify]]:MTP 模块出草稿,主模型一次前向验证,接受规则 $\min(1,p/q)$ 保持分布——**MTP 投机不改变 DeepSeek-V3 的输出分布**。因草稿器与主模型同源同分布,$q$ 极接近 $p$,接受率天然高。

## 图

![[spec-mtp自投机.png]]

## 代码

```bash
# ✅ SGLang:为 DeepSeek-V3 开 MTP 自投机(草稿器即模型自带的 MTP 模块)
python -m sglang.launch_server \
  --model deepseek-ai/DeepSeek-V3 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-num-draft-tokens 2 \
  --speculative-eagle-topk 1
# DeepSeek-V3 的 MTP 走 EAGLE 风格的草稿-验证管线;num-steps=1 用 MTP1(接受率>80%)
```

```python
# ✅ vLLM:DeepSeek-V3 MTP 作为投机方法
from vllm import LLM
llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    speculative_config={"method": "deepseek_mtp", "num_speculative_tokens": 1},
)
```

```python
# ❌ 反例：把 MTP 草稿长度开得过大
"num_speculative_tokens": 4    # 远端 MTP 模块(t+4)接受率低
# 远 token 接受率衰减 → 草稿验证白做、净加速反降；MTP1(k=1)往往是甜点
```

`❌` MTP 草稿开太长 = 远端模块接受率低、收益边际递减;`✅` DeepSeek-V3 实践多用 **MTP1(k=1)** 这个甜点(接受率 >80%、~1.8×),要更长草稿需评估远端模块接受率。


![[spec-077MTP结构对比.png]]

## 面试高频

- **Q:MTP 最初是为什么设计的?** 是 DeepSeek-V3 的**训练目标**——多 token 预测致密训练信号、迫使模型规划未来,**提升主模型本身质量**;投机解码是它的"副产品红利"。
- **Q:什么是 self-speculation?** 模型自带的 MTP 模块在推理时直接当草稿器,无需独立草稿模型或额外训练的草稿头;草稿器与模型同源、词表对齐、分布一致 → 接受率高。
- **Q:MTP 和 Medusa 的结构区别?** Medusa 各头独立、都只条件于同一 $h_t$;DeepSeek MTP 是**因果链式**——第 $d$ 模块条件于第 $d{-}1$ 模块输出,保持序列依赖,更连贯、接受率更高。
- **Q:DeepSeek-V3 的实测收益?** MTP1(第二 token)接受率 >80%,生成吞吐约 **1.8×**;且因 MTP 是无损投机,不改变输出分布。
- **Q:为什么 MTP 接受率天然高?** 草稿器(MTP 模块)和主模型一起训练、同源同分布,$q$ 极接近 $p$,$\min(1,p/q)$ 接受概率高。

## 关键事实

- **DeepSeek-V3 Technical Report**,arXiv:2412.19437(2024);MTP 模块共 ~14B 参数(11.5B 独有 + 2.5B 共享),每模块 = 共享 embed + 共享 head + 1 Transformer 块 + 投影。
- MTP 双重身份:**训练目标**(提升主模型质量)+ **推理草稿器**(self-speculation 无损加速)。
- 结构:**因果链式**(第 $d$ 模块条件于第 $d{-}1$),区别于 Medusa 的并行独立头;更连贯、接受率更高。
- 收益:**MTP1 接受率 >80%**,生成吞吐 ≈ **1.8×**;SGLang/vLLM 2025 均内置 DeepSeek-V3 MTP 投机路径。
