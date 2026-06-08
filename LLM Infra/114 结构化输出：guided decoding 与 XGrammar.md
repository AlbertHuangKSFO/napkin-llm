[[114 结构化输出：guided decoding 与 XGrammar]] 是**在解码每一步用一台「语法自动机」算出合法 token 集合,把所有非法 token 的 logit 置为 -∞,从而强制模型只能吐出符合 JSON Schema、正则或上下文无关文法(CFG)的输出**。它把「让模型输出合法 JSON」从「祈祷 + 重试 + 正则修补」变成**结构上不可能出错**:无效 token 根本采样不到。这是 [[118 函数调用与工具服务化|函数调用服务]] 的底层引擎(工具调用本质就是「输出必须匹配某个工具参数的 schema」),并与采样紧密耦合——掩码作用在 logits 上,采样发生在掩码之后。代表实现:**XGrammar**(下推自动机 + token 掩码缓存)、**Outlines**(正则/JSON 编成有限状态机)。引擎侧:vLLM 的 `guided_json` / `guided_regex` / `guided_grammar`。

## 直觉类比
把生成想成在**迷宫**里走:模型每步想往某个方向迈(logits 给出偏好),而**文法**是迷宫的墙——有些方向是死路(非法 token)。Guided decoding 就是在每个路口先看墙(算 mask),把所有撞墙的方向画上 ✗,模型只能在剩下的合法出口里按偏好挑一个。没有墙时(自由文本)它想去哪去哪;一旦进了 `{"name":` 这种强约束段,出口可能只剩一个,直接被推着走(这就是 **jump-ahead**)。

## 小数字例子
设词表 5 万。当前正在生成 JSON,刚吐完 `{` ,文法规定下一个**只能是** `"` 或空白:
- 合法 token 假设 12 个,**非法 49988 个**全部 logit = -∞。
- 模型原本可能想吐 ` the`(自然语言习惯),被屏蔽 → 改吐 `"`,绝不会产生非法 JSON。
- **jump-ahead**:若 schema 强制接下来必为字面量 `"name":`,引擎**直接把这 4 个 token 拼进序列**,跳过 4 次前向 → 这几步零解码开销。
- **澄清:跳的是「逐步解码」不是「全部计算」**。jump-ahead 省掉的是这 4 个 token 各自一次「前向→算 logits→采样」的串行循环——既然续写唯一确定,采样毫无意义。但这 4 个 token 的 **KV 仍要算**:引擎把它们当作已知 prefix,用**一次** prefill 式前向(类似处理 prompt)把 4 个位置的 K/V 并行填进 cache,而非零计算。省的是 $4\times$ decode 步的串行延迟与采样,换来 $1\times$ 批量 prefill,所以快、但不是免费。
- 成本:每步要算一次 5 万维掩码。XGrammar 实测掩码生成与 GPU 前向**重叠**后近乎零开销,相比早期纯 CPU 实现端到端最高约 **80×** token 速率。

## 原理:下推自动机与 token 掩码
文法被编成一台**字节级下推自动机(PDA)**——比有限状态机多一个**栈**,因此能表达 JSON 的嵌套/递归(`{` 入栈、`}` 出栈)。第 $t$ 步,自动机当前状态 $q_t$(含栈)决定一个合法 token 集合 $\mathcal{V}_{\text{valid}}(q_t)\subseteq \mathcal{V}$。掩码作用在 logits 上:

$$
\tilde{\ell}_{t,i}=
\begin{cases}
\ell_{t,i}, & i\in\mathcal{V}_{\text{valid}}(q_t)\\[2pt]
-\infty, & \text{否则}
\end{cases}
\qquad
p_t=\mathrm{softmax}(\tilde{\ell}_t)
$$

非法项经 softmax 后概率为 0,采样必落在合法集合。关键优化:XGrammar 把 token 分两类——**上下文无关 token**(只看自动机局部位置就能判定,占通常 **>99%**,可**预计算并缓存**)与**上下文相关 token**(需看完整栈状态,运行时才算)。于是绝大多数掩码查表即得,只有少数实时计算。

三类约束(正则 / JSON Schema / CFG)对应两种自动机:扁平结构 FSM 够用,递归嵌套必须 PDA(带栈)——这也是 XGrammar 一套通吃的根因:

![[srv-114三类约束对比.png]]

![[srv-guided-decoding下推自动机.png]]

逐步看「掩码 + jump-ahead」如何在每个解码步把非法 token 钉成 -∞、并在唯一续写时直接跳步:

![[srv-114约束解码时序.png]]

## 配置 / 代码
```python
# vLLM:guided_json / guided_regex / guided_grammar
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# 1) 直接给 JSON Schema(后端默认 XGrammar)
schema = {
    "type": "object",
    "properties": {"city": {"type": "string"},
                   "temp_c": {"type": "number"}},
    "required": ["city", "temp_c"],
}
sp = SamplingParams(
    temperature=0.7,
    guided_decoding=GuidedDecodingParams(json=schema),
)

# 2) 正则约束(Outlines 风格)
sp_re = SamplingParams(
    guided_decoding=GuidedDecodingParams(regex=r"\d{4}-\d{2}-\d{2}"))

# 3) CFG / EBNF 文法(需 PDA,XGrammar)
ebnf = r'''
root   ::= "(" expr ")"
expr   ::= term ("+" term)*
term   ::= [0-9]+
'''
sp_g = SamplingParams(
    guided_decoding=GuidedDecodingParams(grammar=ebnf))

out = llm.generate(["上海现在多少度?以 JSON 回答。"], sp)
print(out[0].outputs[0].text)   # 保证是合法 JSON
```

```text
❌ 不约束:prompt 里写「只输出 JSON」,靠模型自觉 → 偶发多说一句、漏引号、尾逗号 → 下游 json.loads 崩,只能正则修补 + 重试,长尾延迟与失败率不可控
✅ guided_json:无效 token 直接屏蔽,输出结构上必然合法,一次成,无需重试
```

## 面试高频
- **guided decoding 怎么保证输出合法?** 每步用文法自动机算合法 token 掩码,非法 token logit 置 -∞ 再 softmax,采样只可能落在合法集合;不是事后校验,是**生成时就不可能越界**。
- **XGrammar 和 Outlines 区别?** Outlines 把正则/JSON 编成**有限状态机(FSM)**,逐状态查表;XGrammar 用**下推自动机(带栈)**,能表达完整 CFG 的**嵌套递归**,并把 token 分上下文无关(>99%,预缓存)/相关两类做加速。
- **为什么早期实现慢、XGrammar 快在哪?** 慢在每步都要对 5 万词表实时算掩码且在 CPU 慢路径;XGrammar 预计算缓存大部分掩码 + 与 GPU 前向重叠,掩码近乎零开销。
- **什么是 jump-ahead?** 当文法此刻只允许唯一续写(如固定字段名),直接把这些确定 token 拼进序列,跳过多次前向,降时延。
- **约束解码有什么副作用?** 文法太严会把模型逼进低概率分支 → **内容质量下降**;若分词与文法不一致可能出现「mask 全 0」无解;还要保证 schema 本身可满足。
- **和函数调用什么关系?** 工具调用 = 「输出必须匹配工具参数 schema」,`tool_choice=required` 时引擎正是用 guided decoding 强制产出合法调用。

## 关键事实
- **XGrammar**,MLC/CMU,arXiv **2411.15100**(2024-11):字节级 PDA + 自适应 token 掩码缓存;上下文无关 token 占 >99% 可预计算;集成 vLLM/SGLang/MLC-LLM,相比早期方案端到端最高约 **80×** 输出 token 速率,已成多引擎默认后端。
- **Outlines**,dottxt,把正则/JSON Schema 预编译成 FSM 做约束生成(2023 起);早于 PDA 方案,擅长正则/扁平 JSON,递归 CFG 表达力弱于 PDA。
- vLLM 提供 `guided_json` / `guided_regex` / `guided_grammar` / `guided_choice`,后端可选 XGrammar(默认)、Outlines、lm-format-enforcer(2025 现状)。
- 适用:[[118 函数调用与工具服务化|函数调用]]、抽取式 RAG、Agent 工具参数、固定 schema API 输出;均靠同一套约束解码基础设施。
