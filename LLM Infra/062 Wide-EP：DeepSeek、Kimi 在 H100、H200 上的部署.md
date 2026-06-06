这是 [[LLM Infra]] 域里 [[061 专家并行 EP：大规模 MoE 服务|专家并行 EP]] 的工业化形态。**Wide-EP(宽专家并行)**指把 EP 度从机内几张卡一路拉到**几十甚至上百张卡**,跨多个节点,同时配合 [[048 为何分离 prefill 与 decode|PD 分离]]、[[063 注意力数据并行(attn-DP)|attn-DP]]、[[068 all-to-all：MoE、EP 的通信瓶颈|all-to-all]] 的低延迟内核,把 [[LLM/047 DeepSeek MoE：细粒度与共享专家|DeepSeek MoE]] 这类巨型 MoE 服务到极限吞吐。核心直觉:EP 度越大,**每张卡分到的专家越少(权重越薄)**,而把许多卡的 token 聚到同一专家上又能**喂出足够大的 batch**,让稀疏 MoE 的 GEMM 不再因 batch 太小而饿死算力。它是 2025 年 DeepSeek / Kimi 等开源模型生产部署的事实标准。

把它想成连锁餐厅的中央厨房改造。小 EP 像每家店自带全套厨师——专精度低、设备重复。Wide-EP 则把全城的某一道菜的订单都汇到一个超大中央厨房工位(一个专家占一卡),订单量(batch)够大才值得开足马力的设备;同时把"备菜"(prefill)和"出餐"(decode)拆成两个独立车间各自扩缩——备菜要大锅猛火(算力受限,大 batch),出餐要快(访存受限,低延迟)。两套车间用 EP 度、专家摆位都不同。

小数字感受(DeepSeek-V3 官方推理系统,2025)。**Prefill**:最小单元 4 节点 32 卡,**EP32**,每卡 9 个路由专家 + 1 共享专家 + 32 个冗余专家,目的是让每专家 batch 足够大。**Decode**:最小单元 40 节点 320 卡,**EP320**,每卡只放 **1 个专家**,另有 64 卡承载冗余/共享专家——decode 把权重摊到几乎不占显存,显存全留给 KV 与并发。LMSYS 在 96×H100 上复现:约 **52.3K 输入 + 22.3K 输出 token/s/节点**(2K 输入),较 vanilla TP 最高约 **5×** 输出吞吐。Kimi K2(384 专家)在 128×H200 上约 **\$0.21 / 1M 输出 token**(短上下文)。

$$
\text{每卡专家数}=\frac{N_{\text{expert}}+N_{\text{redundant}}}{\text{EP}},\qquad
\overline{\text{每专家 token}}=\frac{B_{\text{global}}\cdot k}{N_{\text{expert}}}
$$

Wide-EP 的逻辑:EP↑ ⇒ 每卡专家数↓(权重薄) 且 聚合全局 batch $B_{\text{global}}$↑ ⇒ 每专家 token↑(GEMM 满)。瓶颈从显存 / 算力转移到 **all-to-all 通信 + 专家负载不均**。

![[ipar-WideEP部署.svg]]

```python
# SGLang:Wide-EP 部署 DeepSeek-V3,PD 分离 + EPLB(2025 形态)
# Prefill 节点(大 batch、高吞吐 DeepEP):
#   python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 \
#     --disaggregation-mode prefill --tp 32 --enable-ep-moe --ep-size 32 \
#     --enable-eplb            # 专家并行负载均衡:周期性重排 + 冗余专家

# Decode 节点(低延迟 DeepEP、超大 EP):
#   python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 \
#     --disaggregation-mode decode --enable-ep-moe --ep-size 320 \
#     --enable-dp-attention    # 注意力走 DP,避免 MLA KV 在 TP 各卡复制
```

```text
❌ 把 EP 度无脑拉满却不开 EPLB / 不做通信计算重叠
   → 少数热门专家的卡成为整层短板(木桶效应),all-to-all 暴露在关键路径上
✅ 大 EP + EPLB(冗余/复制热门专家)+ dual-batch overlap + DeepEP 低延迟内核
   → 负载拉平、通信被计算掩盖,才真正吃到 Wide-EP 的吞吐
```


![[ipar-062WideEP-EPLB均衡.svg]]
![[ipar-062WideEP-PD车间.svg]]

## 面试高频
- **Wide-EP 为什么必须配 PD 分离?** prefill 算力受限、要大 batch、用高吞吐 all-to-all 内核;decode 访存受限、要低延迟、用低延迟内核且把专家摊到极致(每卡 1 专家)。两者最优 EP 度与专家摆位完全不同,聚在一起会互相拖累,所以拆成独立服务各自扩缩。
- **EP 度越大,最大的新问题是什么?** 专家负载不均被放大 + all-to-all 跨节点跳数变多。前者用 EPLB(冗余专家、周期再均衡)解,后者靠 DeepEP/NVSHMEM 内核 + 通信计算重叠(dual-batch overlap)掩盖。
- **为什么 decode 阶段每卡只放 1 个专家?** decode 的 batch 小、是访存受限,把专家摊到 320 卡能让权重几乎不占显存、显存全留给 KV 和并发,同时每张卡专家权重读取压力最小。
- **冗余专家(redundant experts)是什么?** 按线上统计把热门专家在多卡复制,均摊其流量,降低单卡热点——本质是 EPLB 的手段。

## 关键事实
- DeepSeek-V3 官方推理系统(**2025** 开源周):Prefill EP32(4 节点/32 卡,每卡 9 路由 + 1 共享 + 32 冗余专家);Decode EP320(40 节点/320 卡,每卡 1 专家,64 卡放冗余/共享)。
- LMSYS(**2025-05**)在 96×H100 复现 DeepSeek:PD 分离 + 大 EP,约 52.3K 输入 / 22.3K 输出 token/s/节点;decode EP72 较 TP16 约 5.2×。
- LMSYS(**2025-07**)Kimi K2(384 专家)在 128×H200:约 \$0.21 / 1M 输出 token(短上下文),用 EPLB + SGLang Router。
- Red Hat llm-d(**2025-09**)与 vLLM(**2025-12**,约 2.2K tok/s/H200)将 Wide-EP 做成 K8s 原生 well-lit path:DeepEP 内核(高吞吐/低延迟双模)+ EPLB + dual-batch overlap。
