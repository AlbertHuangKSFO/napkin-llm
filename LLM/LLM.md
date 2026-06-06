# LLM 学习地图

> 从 Transformer 的每一个零件「为何这样设计」,一路讲到「能从头训练一个会说话的 tiny GPT」。承接 [[深度学习基础]](数学→神经网络→RNN→注意力起源),这一域展开:**Transformer 架构逐部件** → **注意力变体全谱**(MQA/GQA/MLA/线性/FlashAttention/Mamba)→ **位置编码专题**(RoPE 及外推)→ **发展史与现代架构** → **MoE** → **分词与嵌入** → **预训练** → **分布式训练** → **参数量与算力** → **后训练与对齐**(SFT/RLHF/DPO/GRPO/R1)→ **量化与压缩** → **推理与服务** → **评估** → **从零实现**。每篇按「直觉→小数字例子→公式推导→图→可运行代码→面试高频→关键事实(带出处)」展开,面向零基础又直接对标 MLE 面试。这里只放定位与链接,不写正文。

## ① Transformer 架构逐部件 · 为何这样设计
- [[001 Transformer 总览：为何抛弃 RNN|Transformer 总览]]
- [[002 自注意力 Self-Attention|自注意力 Self-Attention]]
- [[003 Query、Key、Value 的设计|Query、Key、Value 的设计]]
- [[004 缩放点积注意力(为何除以根号 dk)|缩放点积注意力]]
- [[005 多头注意力 Multi-Head|多头注意力]]
- [[006 注意力的矩阵形式与张量形状|注意力的矩阵形式与张量形状]]
- [[007 因果掩码与 padding 掩码|因果掩码与 padding 掩码]]
- [[008 前馈网络 FFN(为何 4 倍、为何两层)|前馈网络 FFN]]
- [[009 残差连接与梯度流|残差连接与梯度流]]
- [[010 层归一化：Pre-LN 与 Post-LN|层归一化：Pre-LN vs Post-LN]]
- [[011 Encoder-Decoder、Decoder-Only 与 Encoder-Only|三类架构]]
- [[012 交叉注意力 Cross-Attention|交叉注意力]]
- [[013 Transformer 整体数据流(逐张量形状)|整体数据流]]
- [[014 注意力复杂度 O(n²) 与瓶颈|注意力复杂度 O(n²)]]
- [[015 Transformer 训练稳定性|训练稳定性]]
- [[016 输出层、tied embedding 与 logits|输出层与 tied embedding]]
- [[017 Dropout 在 Transformer 中的位置|Dropout 的位置]]

## ② 注意力变体与高效注意力
- [[018 MQA 多查询注意力|MQA 多查询注意力]]
- [[019 GQA 分组查询注意力|GQA 分组查询注意力]]
- [[020 MLA 多头潜在注意力(DeepSeek)|MLA 多头潜在注意力]]
- [[021 局部与滑窗注意力(Longformer、Mistral SWA)|局部与滑窗注意力]]
- [[022 稀疏注意力(BigBird、块稀疏)|稀疏注意力]]
- [[023 线性注意力(Linear Transformer、Performer)|线性注意力]]
- [[024 Linformer 与低秩近似|Linformer 与低秩近似]]
- [[025 FlashAttention(IO 感知精确注意力)|FlashAttention]]
- [[026 PagedAttention 与 KV 分页|PagedAttention 与 KV 分页]]
- [[027 状态空间模型与 Mamba|状态空间模型与 Mamba]]

## ③ 位置编码专题
- [[028 位置编码总览：为何 Transformer 需要|位置编码总览]]
- [[029 正弦余弦位置编码(推导)|正弦余弦位置编码]]
- [[030 可学习与相对位置编码|可学习与相对位置编码]]
- [[031 RoPE 旋转位置编码(推导与实现)|RoPE 旋转位置编码]]
- [[032 RoPE 外推：NTK-aware、位置插值、YaRN|RoPE 外推]]
- [[033 ALiBi 与 NoPE|ALiBi 与 NoPE]]

## ④ 发展史与现代架构
- [[034 发展史总览：2017 到今|发展史总览]]
- [[035 BERT：双向编码与 MLM|BERT]]
- [[036 GPT 系列：自回归与规模化|GPT 系列]]
- [[037 T5 与 BART：去噪 encoder-decoder|T5 与 BART]]
- [[038 LLaMA 架构解剖|LLaMA 架构解剖]]
- [[039 Mistral、Qwen、DeepSeek 架构选择|Mistral、Qwen、DeepSeek 架构选择]]
- [[040 现代 decoder-only 配方汇总|现代 decoder-only 配方]]
- [[041 多模态 LLM 架构(ViT、CLIP、投影对齐)简介|多模态 LLM 架构]]

## ⑤ MoE 专题
- [[042 MoE 动机：稀疏激活与容量解耦|MoE 动机]]
- [[043 门控路由与 top-k 选择|门控路由与 top-k]]
- [[044 专家容量、丢弃与负载均衡损失|专家容量与负载均衡]]
- [[045 Switch Transformer 与 GShard|Switch 与 GShard]]
- [[046 Mixtral 稀疏 MoE|Mixtral 稀疏 MoE]]
- [[047 DeepSeek MoE：细粒度与共享专家|DeepSeek MoE]]
- [[048 路由稳定性：router z-loss|路由稳定性 router z-loss]]
- [[049 专家并行 EP 与 MoE 部署|专家并行 EP 与部署]]

## ⑥ 分词与嵌入
- [[050 分词总览与子词动机|分词总览与子词动机]]
- [[051 BPE 与 Byte-level BPE|BPE 与 Byte-level BPE]]
- [[052 WordPiece、Unigram 与 SentencePiece|WordPiece、Unigram、SentencePiece]]
- [[053 词表、特殊 token 与对话模板|词表、特殊 token 与对话模板]]
- [[054 词嵌入层与权重绑定|词嵌入层与权重绑定]]
- [[055 分词的坑：数字、代码、多语言与 token 攻击面|分词的坑]]

## ⑦ 预训练
- [[056 预训练目标：自回归、掩码与去噪|预训练目标]]
- [[057 数据：清洗、去重与质量过滤|数据清洗、去重、过滤]]
- [[058 数据配比与课程|数据配比与课程]]
- [[059 tokenizer 训练|tokenizer 训练]]
- [[060 训练目标与 loss 实现|训练目标与 loss 实现]]
- [[061 优化器与超参(AdamW)|优化器与超参]]
- [[062 学习率调度：warmup 加 cosine 与 WSD|学习率调度 warmup/cosine/WSD]]
- [[063 批大小、梯度累积与 critical batch size|批大小与 critical batch size]]
- [[064 混合精度 FP16、BF16 与 FP8|混合精度 FP16/BF16/FP8]]
- [[065 梯度检查点与激活重计算|梯度检查点与重计算]]
- [[066 训练不稳定：loss spike 与对策|loss spike 与对策]]
- [[067 长上下文训练与续训|长上下文训练与续训]]

## ⑧ 分布式训练
- [[068 并行总览：DP、TP、PP、EP、SP|并行总览]]
- [[069 数据并行与 AllReduce|数据并行与 AllReduce]]
- [[070 ZeRO 与 FSDP|ZeRO 与 FSDP]]
- [[071 张量并行(Megatron)|张量并行 Megatron]]
- [[072 流水线并行与气泡|流水线并行与气泡]]
- [[073 序列并行与上下文并行|序列并行与上下文并行]]
- [[074 通信原语与计算通信重叠|通信原语与重叠]]

## ⑨ 参数量与算力
- [[075 参数量逐层手算(GPT 全拆)|参数量逐层手算]]
- [[076 显存占用估算(参数、梯度、优化器、激活、KV)|显存占用估算]]
- [[077 训练 FLOPs 与 6ND 法则|训练 FLOPs 与 6ND]]
- [[078 推理算力、吞吐与延迟、Roofline|推理算力与 Roofline]]
- [[079 Scaling Law 与 Chinchilla 最优|Scaling Law 与 Chinchilla]]

## ⑩ 后训练与对齐
- [[080 后训练总览：SFT 到 RM 到 RLHF|后训练总览]]
- [[081 指令微调 SFT 与数据构造|指令微调 SFT]]
- [[082 偏好数据与标注|偏好数据与标注]]
- [[083 奖励模型 RM|奖励模型 RM]]
- [[084 策略梯度与 PPO 基础|策略梯度与 PPO]]
- [[085 RLHF 全流程与 KL 约束、奖励黑客|RLHF 全流程]]
- [[086 DPO 直接偏好优化(推导)|DPO 直接偏好优化]]
- [[087 DPO 家族：IPO、KTO、ORPO、SimPO|DPO 家族]]
- [[088 GRPO 与可验证奖励|GRPO 与可验证奖励]]
- [[089 推理模型与 RL：o1、R1 的长 CoT 与自我反思|推理模型与 RL(o1、R1)]]
- [[090 RLAIF、宪法 AI 与过程奖励 PRM|RLAIF、宪法 AI、PRM]]
- [[091 高效微调：LoRA、QLoRA、Adapter、Prefix|高效微调 LoRA/QLoRA]]

## ⑪ 量化与压缩
- [[092 量化基础：对称非对称、per-tensor、per-channel、per-group|量化基础]]
- [[093 PTQ 与 QAT|PTQ 与 QAT]]
- [[094 LLM.int8 与离群值|LLM.int8 与离群值]]
- [[095 GPTQ|GPTQ]]
- [[096 AWQ 与 SmoothQuant|AWQ 与 SmoothQuant]]
- [[097 NF4 与 QLoRA 4-bit|NF4 与 QLoRA 4-bit]]
- [[098 FP8 训练推理与 AQLM 极低比特|FP8 与 AQLM]]
- [[099 剪枝、稀疏化与蒸馏压缩|剪枝、稀疏化与蒸馏]]

## ⑫ 推理与服务
- [[100 解码策略：贪心与 Beam|解码策略：贪心与 Beam]]
- [[101 采样解码：温度、top-k、top-p、min-p、重复惩罚|采样解码]]
- [[102 KV-Cache|KV-Cache]]
- [[103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV-Cache 优化]]
- [[104 投机解码与 Medusa、Lookahead|投机解码]]
- [[105 连续批处理 continuous batching|连续批处理]]
- [[106 chunked prefill 与 prefill、decode 解耦|chunked prefill 与 PD 解耦]]
- [[107 长上下文推理：YaRN、位置插值与 StreamingLLM|长上下文推理]]
- [[108 推理引擎：vLLM、TensorRT-LLM、llama.cpp、SGLang|推理引擎]]

## ⑬ 评估
- [[109 语言模型评估：困惑度与 bits-per-byte|语言模型评估]]
- [[110 下游基准：MMLU、GSM8K、HumanEval、MT-Bench|下游基准]]
- [[111 评估陷阱：数据污染与格式敏感|评估陷阱]]
- [[112 人评、LLM-as-judge 与 Arena|人评、LLM-as-judge、Arena]]

## ⑭ 从零实现 capstone(可运行)
- [[113 从零实现总览：课程地图到代码|从零实现总览]]
- [[114 手写自动微分引擎(micrograd 级)|手写自动微分引擎]]
- [[115 手写多头注意力与 Transformer Block(numpy)|手写注意力与 Transformer Block]]
- [[116 实现 BPE 分词器|实现 BPE 分词器]]
- [[117 训练一个 tiny GPT(PyTorch,可跑)|训练 tiny GPT]]
- [[118 采样生成与困惑度评估|采样生成与困惑度评估]]
- [[119 给 tiny GPT 做 SFT 与 LoRA(迷你对齐)|tiny GPT 做 SFT 与 LoRA]]

## ⑮ 跨域桥接
- 上游地基:整域 [[深度学习基础]] —— 自注意力承接 [[60 注意力机制的起源(Bahdanau、Luong)|注意力起源]];残差/归一化承接 [[52 残差连接与深度可训练性|残差]]、[[43 归一化(BatchNorm、LayerNorm、RMSNorm、GroupNorm)|归一化]];从零实现承接 [[20 反向传播的数学推导|反向传播]]、[[18 计算图(前向)|计算图]]。
- 词嵌入 ↔ 检索:[[054 词嵌入层与权重绑定|词嵌入]] → [[04 Embedding 与向量数据库|RAG Embedding]]。
- 后训练 ↔ Agent:[[085 RLHF 全流程与 KL 约束、奖励黑客|RLHF]]、[[088 GRPO 与可验证奖励|GRPO]] → [[32 Agentic RL 与训练|Agentic RL]]。
- 攻击面 ↔ AI 安全:[[055 分词的坑：数字、代码、多语言与 token 攻击面|token 攻击面]]、对齐边界 → [[05 Prompt Injection 提示注入|提示注入]]、[[06 Jailbreak 越狱|越狱]]。
