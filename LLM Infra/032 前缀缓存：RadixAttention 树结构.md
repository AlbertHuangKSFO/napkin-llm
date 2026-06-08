[[032 前缀缓存：RadixAttention 树结构|前缀缓存：RadixAttention 树结构]]把 [[LLM/102 KV-Cache|KV-Cache]] 组织成一棵 **radix tree(基数树)**:所有请求的前缀 KV 挂在树上,**共享前缀**(同一 system prompt、同一 few-shot、同一多轮对话历史)落在同一节点、**只存一份并跨请求复用**,显存满时按 **LRU** 从叶子驱逐。它是 SGLang 的核心,把 [[LLM/026 PagedAttention 与 KV 分页|PagedAttention]] 的 copy-on-write 共享系统化成可自动匹配的树,是降 TTFT、增吞吐的关键 [[LLM/103 KV-Cache 优化：GQA、量化、驱逐与前缀共享|KV 优化]]。

## 直觉

很多请求开头**长得一模一样**:同一段 system prompt、同一组 few-shot 例子、多轮对话的历史。老式系统每条请求都把这段前缀**重算一遍 prefill、各存一份 KV**——纯浪费。RadixAttention 把"前缀"当成树上的**路径**:root → "You are helpful..." → few-shot → 各请求独有的问题。公共路径的 KV 只算一次、只存一次;新请求来了沿树**匹配最长公共前缀**,命中部分直接复用 KV(跳过那段 prefill),只算独有尾部。基数树(radix tree)是把单链 trie 压缩的省空间变体,边上标的是**一段** token 而非单个,适合长前缀。

**生活类比**:很多人点同一家奶茶,开头都用同一锅「半成品茶底」(共享 system prompt + few-shot),只是后面各加各的料——顾客 1 加珍珠、顾客 2 加椰果、顾客 3 加布丁。聪明的店不会每杯都重熬一锅茶底:茶底**只熬 1 次、只占 1 锅**,做每杯时沿着「茶底→加料」这条路径走,公共的茶底直接复用,只调自己那点料。算笔账:100 杯都用一段 800 token 的茶底、各接 50 token 的料。每杯各熬一锅 = 100×850=85000 token 全算,茶底白熬 100 遍;茶底共享 = 800 + 100×50=5800 token,**省约 93%**,茶底 KV 从 100 份缩成 1 份。这就是树结构:大家共用一条主干(茶底),到分叉点才各算各的尾(加料)——这正是 RadixAttention 的 radix tree。

![[kv-032类比奶茶茶底.png]]

## 例子

100 个请求共享一段 **800 token** 的 system prompt + few-shot,各自再接 ~50 token 的问题:
- **无前缀缓存**:每条都 prefill 850 token,共 $100\times850=85000$ token 的 prefill 计算;前缀 KV 存 100 份。
- **RadixAttention**:前缀 800 token 只算 1 次、存 1 份(树上一条路径);每条只算自己 50 token 尾部。prefill 计算 $\approx 800 + 100\times50 = 5800$ token,**省约 93%**;前缀 KV 显存省 100→1。

命中率随共享程度变,实测 **50%–99%**(Zheng 2024)。命中越高,TTFT 越低、吞吐越高。

## 原理

设共享前缀长 $L_p$,独有尾长 $L_t$,$N$ 个请求。命中前缀后总 prefill token:

$$
T_{\text{prefill}} = L_p + N\cdot L_t \quad(\text{vs 无缓存 } N\cdot(L_p+L_t))
$$

省下比例 $\approx \dfrac{(N-1)L_p}{N(L_p+L_t)}$,前缀越长、共享越广越省。树操作:**插入**(新请求路径并入)、**匹配**(最长公共前缀查找)、**驱逐**(显存满时从叶子 LRU 淘汰,共享前缀因频繁命中常驻)。配 **cache-aware 调度**:把命中同前缀的请求排在一起跑,提高命中率。KV 物理存储仍走 [[030 PagedAttention 深入：KV 当虚拟内存|分页 block]],radix 树管的是"哪些 block 属于哪段前缀、谁在共享"。

## 图

![[kv-radix树共享.png]]

![[kv-Radix三操作.png]]

![[kv-032前缀复用省算手算.png]]

## 代码

```python
# SGLang：RadixAttention 默认开启，前缀自动匹配复用
import sglang as sgl

@sgl.function
def qa(s, question):
    s += sgl.system("You are a helpful assistant. " + FEW_SHOT)  # ✅ 公共前缀
    s += sgl.user(question)                                       # 独有尾部
    s += sgl.assistant(sgl.gen("answer", max_tokens=128))

# 100 个请求共享 system+few-shot：前缀 KV 只算一次、存一份，命中复用
states = qa.run_batch([{"question": q} for q in questions])

# ✅ 把变动内容放尾部、固定内容放前缀 —— 最大化前缀命中
#    prompt = SYSTEM + FEW_SHOT + user_question
# ❌ 反例：把时间戳/请求 ID/随机内容放在前缀最前面
#    prompt = f"[{timestamp}] " + SYSTEM + ...   # 前缀每次都变 → 永不命中
```

`❌` 前缀命中的铁律:**变动内容必须在后,固定内容在前**。任何放在前缀里的易变字段(时间戳、UUID、用户名)都会让整条前缀 hash/路径变化,命中率归零。

## 面试高频

- **RadixAttention 是什么?** 把 KV-Cache 存成 radix tree,**跨请求共享前缀** KV(只算一次、存一份),SGLang 核心(Zheng 2024)。
- **为什么用 radix tree 不用普通 trie?** 基数树压缩单链路径、边标一段 token,长前缀更省空间、匹配更快。
- **省在哪?** 省前缀 **prefill 计算**(降 TTFT)+ 省前缀 **KV 显存**(增 batch → 增吞吐)。
- **驱逐策略?** 显存满时 **LRU** 从叶子淘汰冷节点;共享前缀因频繁命中而常驻;还支持 LFU/FIFO/MRU。
- **和 PagedAttention 的 copy-on-write 关系?** 同思想——共享前缀只存一份;RadixAttention 把它系统化成可自动**最长前缀匹配**的树。
- **典型受益场景?** 长 system prompt、few-shot、多轮对话、批量同模板生成。
- **怎么最大化命中?** 固定内容前置、变动内容后置;cache-aware 调度把同前缀请求排一起。

## 关键事实

- Lianmin Zheng 等,*SGLang: Efficient Execution of Structured Language Model Programs*,2024(NeurIPS 2024,arXiv:2312.07104);RadixAttention 公布于 LMSYS 博客 2024-01-17。
- 机制:KV-Cache 存为 **radix tree**,支持前缀的高效查找/插入/驱逐;**LRU** 驱逐 + cache-aware 调度。
- 效果:实测**命中率 50%–99%**,直接转化为更高吞吐、更低 TTFT;前缀越长、共享越广收益越大。
- 与 [[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention]] 互补:分页管物理存储,radix 树管前缀共享与复用;两者都不改注意力数学。
