这是 [[LLM Infra]] 域里把 [[061 专家并行 EP：大规模 MoE 服务|EP]] 真正跑起来的关键拼图,与 [[062 Wide-EP：DeepSeek、Kimi 在 H100、H200 上的部署|Wide-EP]] 配套。**注意力数据并行(attention data parallelism, attn-DP)**只在 [[LLM/020 MLA 多头潜在注意力(DeepSeek)|MLA]] 这类模型上才有强动机:让**注意力层按请求做 [[060 数据并行与副本扩展|数据并行]]**——每张卡持有一份完整的注意力权重、处理不同的请求、各存各的 [[015 KV-Cache 的显存账(逐层手算)|KV-Cache]];而同一批卡的 **MoE 层仍走 EP**。这样组合的根因:MLA 把 KV 压成一个**所有注意力头共享的低秩潜向量**,[[057 张量并行推理：延迟换显存|TP]] 切不动它,只能让每个 TP rank 复制同一份 KV,白白浪费几倍 KV 显存。attn-DP 绕开这个浪费,是 DeepSeek 官方与 vLLM/SGLang 服务 DeepSeek 系的标准方案。

把它想成医院的两道流程用两种排班。**问诊(注意力)**是高度个人化的:每个病人的病史(KV)只跟自己有关,所以让每个全科医生(GPU)各管一批病人、各自保存病史档案最省柜子——这就是按"病人"分(DP)。若强行让两个医生合看同一个病人的同一步(TP),还得各抄一份完整病史(KV 复制),纯浪费。**化验(MoE)**则相反:化验科室成建制(专家)分散,样本(token)按项目送到对应科室(EP)。同一批人、同一批卡,问诊按人切、化验按科切。

**生活类比**:MLA 的 KV 是「一份所有注意力头共享的笔记」(压到约 576 维,很小)。如果按头切(TP=8),这份笔记被全头共用、根本切不开,只能让 **8 张卡每张都抄一整份**——KV 显存 ×8,能服务的并发被压到 1/8,白白重复抄。改成按请求分(attn-DP=8):8 张卡各管 1/8 的病人、**各存各的那份笔记**,KV 总量不变、并发不缩水,谁也不用替别人抄。MoE 那一层仍按科室分(EP),两种切法靠 all-to-all 衔接,常见恒等式 EP = DP × TP。技术对应:共享笔记=MLA 低秩潜向量 KV、抄一份=TP 复制、各管各的病人=按请求 DP。

![[ipar-063类比共享笔记抄几份.svg]]

小数字感受。MLA 把每 token 的 KV 压到约一个 512+64 维的潜向量(远小于 MHA 的几千维)。若注意力走 **TP=8**,这份潜向量 KV 要在 8 张卡上各存一份 → KV 显存 ×8,可服务的并发请求数被压到 1/8;改用 **attn-DP=8**,8 张卡各存自己那 1/8 请求的 KV,**KV 总量不变、并发不缩水**。配合 MoE 的 EP,常见恒等式是 **EP = DP × TP**:注意力 DP 度 × 注意力内 TP 度,正好等于 MoE 的 EP 度,token 在两种切法间用 all-to-all 衔接。

$$
\text{EP}=\text{DP}_{\text{attn}}\times\text{TP}_{\text{attn}}
$$

$$
\text{KV 显存}_{\text{TP}}=\text{TP}\times\text{KV}_{\text{单份}}\quad(\text{复制,浪费})\ \ \text{vs}\ \ \text{KV 显存}_{\text{attn-DP}}=\text{KV}_{\text{单份}}\quad(\text{各存各的})
$$

![[ipar-attnDP组合EP.svg]]

```python
# vLLM:DeepSeek-V3 标准形态 = attn-DP + EP(2025)
# vllm serve deepseek-ai/DeepSeek-V3 \
#   --data-parallel-size 8 \        # 注意力按请求 DP,各卡独立存 KV
#   --enable-expert-parallel        # MoE 自动展开为 EP=DP×TP

# SGLang:等价开关
# python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 \
#   --enable-dp-attention \         # MLA 注意力走 DP,避免 KV 复制
#   --tp 8 --enable-ep-moe --ep-size 8
```

```text
❌ DeepSeek/MLA 模型注意力沿用 dense 的 TP=8
   → MLA 潜向量被全头共享、TP 切不动 → KV 被复制 8 份,并发腰斩到 1/8
✅ 注意力 --enable-dp-attention(按请求 DP),MoE 走 EP
   → KV 不复制、并发不缩;EP = DP × TP 把两层切法对齐
```


![[ipar-063attnDP-KV对比.svg]]

## 面试高频
- **为什么 MLA 注意力不适合 TP?** MLA 的 KV 是一个被**所有注意力头共享的低秩潜向量**,沿头维做 TP 切不开它;只能让每个 TP rank 复制整份潜向量 KV,并重复算 kv_a_proj / kv_b_proj,纯浪费显存与算力。
- **attn-DP 省的到底是什么?** 省 KV-Cache 显存的"复制倍数"。TP 会把同一份 KV 复制 TP 份,attn-DP 让每卡只存自己请求的 KV,并发请求数随之保住(对 MLA 这种 KV 本就小的模型,复制的相对浪费尤其刺眼)。
- **attn-DP 和 EP 怎么衔接?** 注意力 DP 各卡产出本地 token 的激活,经 dispatch all-to-all 送到 MoE 专家所在卡,算完再 combine 回来。常见 **EP = DP × TP**。
- **attn-DP 的代价?** 各 DP 组负载可能不均(请求长度/数量不齐),需要 DP 感知的路由/调度(如 SGLang DP Router)来平衡。

## 关键事实
- MLA(DeepSeek-V2/V3,**2024**)把 KV 压成所有头共享的潜向量,TP 无法切分 → 各 rank 复制 KV,这是 attn-DP 的根本动因。
- DeepSeek-V3(**2024**,arXiv:2412.19437)官方推理即采用"注意力 DP + MoE EP"组合。
- vLLM RFC #16037(**2025**)与 SGLang DP-attention(**2025**)把 attn-DP + EP 做成 DeepSeek 系标准部署;恒等式 EP = DP × TP。
- Perplexity / Red Hat(**2025**)多节点 DeepSeek 部署均报告:开 DP-attention 后 KV 不再被 TP 复制,并发与吞吐显著提升。
