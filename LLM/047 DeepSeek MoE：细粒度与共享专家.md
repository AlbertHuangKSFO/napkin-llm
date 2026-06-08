[[047 DeepSeek MoE：细粒度与共享专家]]:DeepSeek 2024 年提出的 MoE 改良——把专家**切得更细**(每个大专家拆成 $m$ 个小专家、激活更多个)以逼出更高专精度,再**隔离出始终激活的共享专家**专门承载通用知识,从而减少路由专家间的知识冗余,实现"终极专家专精"。

## ① 直觉:常规 MoE 的两个浪费,以及两招对症下药

[[046 Mixtral 稀疏 MoE|常规 MoE]](如 GShard / Mixtral)每层 $N$ 个大专家、选 top-2。DeepSeek 指出它有两个让"专家专精"打折扣的问题:

1. **专家太粗 → 知识混杂**:每个专家是个完整大 [[008 前馈网络 FFN(为何 4 倍、为何两层)|FFN]],被迫往里塞各种不相关的知识,难以"专精"。而且 $N$ 选 2 的组合数有限,灵活性不足。
2. **通用知识被重复存**:很多基础能力(语法、常识)是每个 token 都要用的;常规路由让**每个**专家都得自己存一份这种通用知识 → 大量**冗余**,浪费容量。

DeepSeek 的两招正对这两点:

- **细粒度专家切分(fine-grained segmentation)**:把每个大专家沿 FFN 的中间隐藏维**切成 $m$ 份小专家**,同时把激活数也乘 $m$。总算力、总参数**不变**,但**激活组合数爆炸式增长**,每个小专家只需负责更窄的知识 → 更专精。
- **共享专家隔离(shared expert isolation)**:专门留 $K_s$ 个**共享专家**,**所有 token 都必经**(不经路由)。让它们统一承载通用知识,路由专家就不必各自重复存 → 去冗余,腾出容量去学专业知识。

## ② 例子:切细 + 拆共享,数字对照

**起点**:常规 MoE,16 个专家,每 token 选 top-2。

**第一步:细粒度切分(取 $m=2$)**。把每个专家的 FFN 中间维砍一半、拆成 2 个小专家 → 变成 $16\times2=32$ 个小专家;激活数也 $\times2$ → 选 top-$4$。

- 算力不变:2 个半大专家 ≈ 1 个大专家的算力。
- 灵活性大增:从 $\binom{16}{2}=120$ 种激活组合,涨到 $\binom{32}{4}=35960$ 种 → 路由能更精细地"配方"。

**第二步:共享专家隔离**。从这 32 个里拿出 1 个当**共享专家**(始终激活),剩下 31 个走路由、选 top-3。于是每 token 实际过 = 1 个共享 + 3 个路由 = 4 个小专家(激活总量不变)。

$$
y = \underbrace{\text{FFN}_{\text{shared}}(x)}_{\text{所有 token 必经}} + \sum_{i\in\text{top-3 路由}} g_i\,\text{FFN}_i(x)
$$

DeepSeekMoE 16B 用这套(共 64 个路由专家 + 2 共享、每 token 激活 6 路由 + 2 共享),**仅约 40% 的计算**就追平了稠密 DeepSeek 7B。后来 DeepSeek-V3(671B 总参、37B 激活)把这套推到极致:256 路由专家 + 1 共享、每 token 选 8 路由,每个专家中间维仅 **2048**(非常细粒度)。

**组合数爆炸的直观感受**。激活组合数 = 「从多少专家里选多少个」的组合数 $\binom{mN}{mK}$,决定路由能「配」出多少种不同的专家组合(表达力):

| 方案 | 专家数 | 激活数 | 激活组合数 $\binom{n}{k}$ |
|---|---|---|---|
| Mixtral(粗) | 8 | 2 | $\binom{8}{2}=28$ |
| DeepSeekMoE 16B(细) | 64 | 6 | $\binom{64}{6}\approx 7{\times}10^{7}$ |
| DeepSeek-V3(极细) | 256 | 8 | $\binom{256}{8}\approx 4{\times}10^{14}$ |

同样的总算力,粒度越细、能「配方」的组合越多 → 路由越灵活、专精越可能。这是 DeepSeek 押「细粒度」的数学底气。

![[moe-细粒度共享.png]]

## ③ 原理:细粒度 + 共享专家的公式

设原本 $N$ 个专家、激活 $K$ 个。**细粒度切分**把每个切成 $m$ 份 → $mN$ 个细专家、激活 $mK$ 个;每个细专家的 FFN 隐藏维变为原来的 $1/m$,故总参与总算力不变,但激活组合数从 $\binom{N}{K}$ 增到 $\binom{mN}{mK}$(指数级变大)。

加入 $K_s$ 个**共享专家**后,DeepSeekMoE 层输出:

$$
y = \sum_{s=1}^{K_s}\text{FFN}_s(x)\;+\;\sum_{i=1}^{mN-K_s} g_i\cdot\text{FFN}_i(x)
$$

其中前一项是共享专家(无门控、恒激活),后一项是路由专家,$g_i$ 对**非 top 的路由专家取 0**:

$$
g_i =
\begin{cases}
\text{softmax}_i(\ell) & i\in\text{TopK}(\ell,\,mK-K_s)\\[2pt]
0 & \text{otherwise}
\end{cases}
$$

**为什么共享专家有效?** 通用知识被它"吸收"后,路由专家不再需要人人重复存一份 → 每个路由专家可以把容量全用在专业知识上 → 整体专精度和参数利用率都更高。这与"通用 + 专用"的分工直觉一致。

**均衡问题怎么处理?** 细粒度专家更多 → 更易失衡。DeepSeekMoE 16B 仍用 [[044 专家容量、丢弃与负载均衡损失|辅助负载均衡损失]](含 expert-level 与 device-level 两种);而 DeepSeek-V2/V3 改用 **auxiliary-loss-free** 策略:给每个专家一个**可学习偏置 $b_i$**,只用于路由打分 $\ell_i + b_i$ 的 top-k 选择(不进入加权),根据各专家近期负载动态加减 $b_i$ 来均衡,避免辅助损失对主任务梯度的干扰。

**无辅助损失偏置怎么动(机制细节)**。每一步统计各专家收到的 token 数:**过载的专家把它的 $b_i$ 调小**(下一步更难被选中)、**饿死的专家把 $b_i$ 调大**(更容易被选中)。注意 $b_i$ **只参与 top-k 的「选谁」**,不参与最终加权 $g_i$(加权仍用原始 softmax 分),所以它纠偏路由分布、却不扭曲专家输出的权重。这避免了辅助损失「往主任务里塞一个和效果无关的梯度」的副作用——DeepSeek-V3 报告它在均衡和效果上都优于辅助损失。

![[moe-DeepSeek偏置闭环.png]]

![[moe-门控topk.png]]

## ④ 代码:DeepSeekMoE 层(共享 + 细粒度路由)

```python
import torch, torch.nn as nn, torch.nn.functional as F

class DeepSeekMoE(nn.Module):
    def __init__(self, d, h_fine, n_routed=63, n_shared=1, top_k=3):
        super().__init__()
        # h_fine = 原大专家隐藏维 / m  → 细粒度:每个专家更小、数量更多
        self.shared = nn.ModuleList(_ffn(d, h_fine) for _ in range(n_shared))
        self.routed = nn.ModuleList(_ffn(d, h_fine) for _ in range(n_routed))
        self.gate = nn.Linear(d, n_routed, bias=False)
        self.k = top_k

    def forward(self, x):                              # x:(tokens, d)
        # ✅ 共享专家:所有 token 必经,承载通用知识(不走门控)
        y = sum(s(x) for s in self.shared)
        # ✅ 路由专家:细粒度 + top-k,只算选中的
        logits = self.gate(x)
        val, idx = logits.topk(self.k, dim=-1)
        w = F.softmax(val, dim=-1)                     # 仅在选中路由专家上归一
        for slot in range(self.k):
            for e in idx[:, slot].unique():
                m = idx[:, slot] == e
                y[m] += w[m, slot, None] * self.routed[e](x[m])
        return y

def _ffn(d, h):
    return nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, d))

# ❌ 反面:不拆共享 → 每个路由专家被迫各存一份通用知识,冗余、专精度低
# ❌ 反面:专家粗(h_fine 取整个大隐藏维)→ 激活组合少、知识混杂、难专精
# ✅ 细粒度 + 共享:组合数爆炸 + 去冗余 → 同算力下专精度与利用率更高
```

## 面试高频

- **Q:DeepSeekMoE 两个核心创新是什么?** ①**细粒度专家切分**:把大专家拆成 $m$ 份小专家、激活数 $\times m$,同算力下激活组合数指数级增加 → 更专精；②**共享专家隔离**:留若干恒激活的共享专家承载通用知识 → 去路由专家间的冗余。
- **Q:细粒度切分会增加算力/参数吗?** 不会。隐藏维砍成 $1/m$、数量 $\times m$、激活数 $\times m$,总参与总 FLOPs 不变,只是粒度更细、组合更灵活。
- **Q:共享专家为什么能去冗余?** 通用知识集中由共享专家承载,所有 token 必经它;路由专家就不必各自重复存通用知识,容量全留给专业知识。
- **Q:专家变多更容易失衡,怎么办?** DeepSeekMoE 16B 用 expert/device-level [[044 专家容量、丢弃与负载均衡损失|辅助损失]];V2/V3 用**无辅助损失**的可学习路由偏置 $b_i$(只影响 top-k 选择)动态均衡。
- **Q:和 [[046 Mixtral 稀疏 MoE|Mixtral]] 比?** Mixtral 是粗粒度 8 专家 top-2、无共享专家；DeepSeek 细粒度 + 共享专家 → 更高专精度与参数利用率。DeepSeek-V2 还配 [[020 MLA 多头潜在注意力(DeepSeek)|MLA]]。
- **Q:细粒度为什么提升表达力?** 激活组合数从 $\binom{N}{K}$ 涨到 $\binom{mN}{mK}$(指数级),路由能配出指数级更多的专家组合,而总算力不变。
- **Q:无辅助损失偏置 $b_i$ 怎么工作?** 按各专家近期负载动态调:过载减 $b_i$、饿死加 $b_i$;**只影响 top-k 选择,不影响最终加权**,故纠偏不扭曲输出。
- **Q:DeepSeek-V3 的精确 MoE 配置?** 每层 1 共享 + 256 路由专家、每 token top-8、专家中间维 2048、每 token ≤4 节点、671B/37B、无辅助损失均衡。
- **Q:共享专家会不会也失衡?** 不会——它不走门控、所有 token 必经,负载天然均匀;失衡问题只在路由专家上。

## 关键事实

- 出处:Dai et al.(DeepSeek-AI),*DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*,2024,arXiv:2401.06066(ACL 2024)。
- 两大策略:**细粒度专家切分**(每专家切 $m$ 份、激活 $mK$,总算力不变、激活组合数 $\binom{mN}{mK}$ 指数增长)+ **共享专家隔离**($K_s$ 个恒激活专家承载通用知识、去冗余)。
- DeepSeekMoE 16B(64 路由 + 2 共享,每 token 激活 6 路由 + 2 共享)**仅约 40% 算力**即追平稠密 DeepSeek 7B(同 2T 语料)。
- DeepSeek-V2(2024,arXiv:2405.04434)/V3(2024,arXiv:2412.19437)沿用并放大:V3 为 671B 总参、37B 激活、256 路由专家 + 1 共享、每 token top-8、专家中间维 2048、每 token ≤4 节点;并采用 **auxiliary-loss-free**(可学习专家偏置)负载均衡,14.8T token 预训练。
- 激活组合数:Mixtral $\binom82=28$、DeepSeekMoE 16B $\binom{64}6\approx7\times10^7$、V3 $\binom{256}8\approx4\times10^{14}$——细粒度让组合数指数级膨胀。
- 偏置 $b_i$ 只参与 top-k 选择、不参与加权:过载减、饿死加,纠偏路由分布而不扭曲专家输出权重。
- 配套创新:DeepSeek-V2 同时引入 [[020 MLA 多头潜在注意力(DeepSeek)|MLA]] 压 KV-Cache;部署同样需 [[049 专家并行 EP 与 MoE 部署|专家并行]]。
