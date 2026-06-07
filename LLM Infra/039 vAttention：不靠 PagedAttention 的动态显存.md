[[039 vAttention：不靠 PagedAttention 的动态显存]] 是微软研究院提出的 KV-Cache 显存管理方案:**不改 attention 内核**,改用 **CUDA 虚拟内存管理(VMM)API** 把 KV 的**虚拟地址保持连续**、而**物理页按需分配**——既消除显存碎片,又让 attention kernel 仍看到一段连续 KV。它是 [[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention]](vLLM)的替代路线:PagedAttention 为省碎片把 KV 虚拟布局改成**非连续的分页**,代价是每个 attention 内核都得重写成「感知块表」的版本;vAttention 把「分页」下沉到硬件/驱动的虚拟内存层,让 [[024 FlashAttention 1、2：IO 感知精确注意力|FlashAttention]]、FlashInfer 等**原版内核开箱即用**。

## 直觉类比
你要给一本越写越长的书(KV)预留书架。**PagedAttention** 的做法:把书拆成一页页塞进图书馆各处空格子,再维护一张「**索引卡**(块表)」记每页在哪——省了连续大书架,但任何**读这本书的人(attention kernel)都得先学会查索引卡、跑遍全馆收集散页**。**vAttention** 的做法:借图书馆的「**虚拟书架**」机制——给你一段**看起来连续的虚拟书架编号**,但只有写到的那几格才真去后台挂上实体格子(物理页)。读书的人照常**从头连续翻**,根本不知道背后物理是按需挂的——所以**不用改任何读书人**。

**生活类比**:把一层楼想成一个序列的 KV。vAttention 像「门牌连续编好、房间住到才装修」:一次性把整层门牌 101～108 连号挂好(虚拟地址连续,够 8192 token 用),但只有真住进去的房间才花钱装修(`cuMemMap` 按需挂物理页)。某请求只生成到 4000 token,就只装修 4 间(101~104),剩下 4 间挂着门牌却不占一分钱;房客从 101 顺着往下连续翻,压根不知道后台是按需装修的——所以**任何读书人(attention 内核)都不用改**,原版 FlashAttention/FlashInfer 直接跑。对比 PagedAttention:它把房间打散塞进楼里各处空格,再加一本「索引卡」记每页在哪,虽然也省了连续大房,但每个读书人都得**先学会查索引卡(改写 kernel)**,每出一版新内核就得再适配一遍。vAttention 省掉这支「改装修队」,论文报吞吐最高 ↑1.23×。

![[kv-039类比连续门牌按需装修.png]]

## 小数字例子
一个请求最终生成 4000 token,但事先不知道会多长。
- **静态预分配**(老办法):按最大 8192 token 一次性占满连续显存 → 浪费近一半,且并发数被这预留撑爆。
- **PagedAttention**:按 16-token 块按需分配 → 无碎片,但 FlashAttention 得用 **paged 版本**;每出一个新 attention 内核就要再做一版 paged 适配。
- **vAttention**:预留一段连续**虚拟**地址(够 8192),实际只为已生成的 4000 token **映射物理页**(用 `cuMemMap`),未写到的部分**不占物理显存**。attention 内核看到连续 KV,直接跑原版 FA/FlashInfer。论文报告:相对基于 PagedAttention 的 FA/FlashInfer 内核,吞吐最高 **↑1.23×**。

## 原理:解耦虚拟与物理分配
PagedAttention 的根本副作用:为了运行时分配物理显存,它把 KV 的**虚拟内存布局从连续变成非连续**,于是注意力计算必须显式按块表 gather 散块 → 每个内核都要改。vAttention 用 **CUDA VMM** 把两件事拆开:

$$
\underbrace{\text{虚拟地址}}_{\text{一次预留,连续}}\;\xrightarrow{\ \text{cuMemMap 按需}\ }\;\underbrace{\text{物理页}}_{\text{运行时按 token 增长挂载}}
$$

- **预留连续虚拟地址**:`cuMemAddressReserve` 一段够大的虚拟空间,KV 在虚拟视图里始终连续。
- **按需挂物理页**:`cuMemCreate` + `cuMemMap` 在序列变长时把物理页映射到虚拟地址尾部;请求结束 `cuMemUnmap` 释放。
- **结果**:物理上无碎片(只挂用到的页),虚拟上连续(kernel 不需感知分页),attention 内核**零改动**。

代价/工程点:CUDA VMM 的页粒度较粗、映射有开销,vAttention 引入若干 LLM 专属优化(如**预先映射/重叠分配、更细的分配粒度**)来掩盖映射延迟,逼近 PagedAttention 的显存效率而保住内核兼容性。

![[kv-vAttention对比PagedAttention.png]]

## 配置 / 代码
```cpp
// vAttention 核心思路:CUDA 虚拟内存 API 解耦虚拟连续 / 物理按需(概念示意)
CUdeviceptr kv;                                  // KV 的连续虚拟基址
cuMemAddressReserve(&kv, MAX_KV_BYTES, 0, 0, 0); // ① 预留一大段连续虚拟地址(还没物理页)

// ② 序列变长 → 只为新增 token 挂物理页
CUmemGenericAllocationHandle h;
cuMemCreate(&h, PAGE_BYTES, &prop, 0);           // 申请一页物理显存
cuMemMap(kv + offset, PAGE_BYTES, 0, h, 0);      // 映射到虚拟地址尾部
cuMemSetAccess(kv + offset, PAGE_BYTES, &desc, 1);

// ③ attention 内核照常把 kv 当连续缓冲区用 —— FlashAttention / FlashInfer 原版直接跑
flash_attn(q, /*K,V=*/kv, ...);                  // 无需 block_table、无需改 kernel

// ④ 请求结束:cuMemUnmap + cuMemRelease 回收物理页(虚拟地址可复用)
```

```text
❌ PagedAttention:为省碎片把 KV 虚拟布局改成非连续分页 → 每个 attention 内核都要重写成感知块表版
✅ vAttention:CUDA VMM 让虚拟连续 + 物理按需 → 同样无碎片,但 FA/FlashInfer 原版内核开箱即用
```

![[kv-vAttention显存生命周期.png]]

![[kv-039VMM调用序列.png]]

## 面试高频
- **vAttention 和 PagedAttention 区别?** 都做「按需分配 KV 显存、去碎片」。PagedAttention 在**软件层**把 KV 切非连续块、改 attention 内核;vAttention 在**驱动/硬件虚拟内存层**保持虚拟连续、物理按需,**不改内核**。
- **为什么「不改内核」很重要?** 每出一个新 attention 内核(FA2/FA3/FlashInfer/各模型变体)都要再做一版 paged 适配,维护成本高;vAttention 让原版直接复用,可移植性强。
- **它靠什么实现?** CUDA 虚拟内存管理 API(`cuMemAddressReserve` 预留连续虚拟地址、`cuMemCreate`/`cuMemMap` 按需挂物理页),解耦虚拟与物理分配。
- **代价是什么?** CUDA VMM 页粒度粗、映射有开销;需 LLM 专属优化(预映射/重叠/细粒度)掩盖延迟,否则可能不如 paged 内核高效。
- **性能如何?** 论文报告相对 PagedAttention 版 FA/FlashInfer 内核吞吐最高 ↑1.23×,且更简单、可移植。

## 关键事实
- **vAttention**,Prabhu / Nayak et al.(Microsoft Research + IISc)**2024**(arXiv 2405.04437,`microsoft/vattention` 开源)。
- 核心:用 **CUDA 虚拟内存管理(VMM)** 解耦 KV 的虚拟分配(连续)与物理分配(按需),消除碎片而不改 attention 内核。
- 对比 [[030 PagedAttention 深入：KV 当虚拟内存|PagedAttention]]:后者把 KV 虚拟布局改成非连续分页、需重写内核;vAttention 保持虚拟连续,FA / FlashInfer 等开箱即用。
- 性能:相对基于 PagedAttention 的 FA / FlashInfer 内核,LLM 服务吞吐最高 **↑1.23×**;更简单、可移植。
- 与 [[024 FlashAttention 1、2：IO 感知精确注意力|FlashAttention]] 等 [[029 何时写自定义 kernel：Nsight 性能分析|attention 内核]] 互补——它解决的是**显存如何分配**,不是注意力如何算。
