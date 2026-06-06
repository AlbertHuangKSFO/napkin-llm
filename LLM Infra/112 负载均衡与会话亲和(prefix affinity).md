这是 [[LLM Infra]] 域决定「请求落到哪张卡」的路由层,是 [[060 数据并行与副本扩展|副本扩展]]的必备配套。**会话亲和 / 前缀亲和(prefix affinity)**=L7 负载均衡器**有状态地**把「带相同前缀(系统提示、对话历史)或同一会话」的请求,尽量路由到**同一个实例**,好让它命中那台机器上已经算好的 [[032 前缀缓存：RadixAttention 树结构|前缀缓存]]。和传统无状态轮询(round-robin)最大的不同:轮询追求负载均匀但会把同前缀打散,导致每台机都重算前缀、缓存命中率≈0;亲和则用缓存命中率换 TTFT 与吞吐。

类比:连锁咖啡店的「记住熟客」。轮询是「来一个排一个,谁空谁接」——但你的复杂定制每次都要重新讲一遍(重算前缀)。亲和是「老顾客固定去 3 号窗口」,3 号店员已经记住你的常点(前缀缓存命中),直接出杯(跳过 prefill)。代价:若所有熟客都涌向 3 号窗口,它会排长队(热点)——这就是亲和与均衡的根本张力。

小数字:1000 个请求共享同一个 2000-token 系统提示。纯轮询下 8 个副本各拿 ~125 个,每个副本第一次见到该前缀都要做一次 2000-token 的 prefill,几乎**零复用**。改成 prefix 亲和:同前缀全路由到副本 1,前缀只 prefill 一次,后续 999 个请求**命中 RadixAttention**、跳过前缀的 prefill。设前缀占请求计算的 70%,亲和把这部分省掉 → 该批 TTFT 与算力近似下降到原来的 ~30%。

$$
\rho_i=\frac{\text{负载}_i}{\text{容量}_i},\quad
\text{CHWBL 准入}:\ \text{选 hash 环上首个满足}\ \rho_i\le(1+\varepsilon)\bar\rho\ \text{的实例}
$$

路由实现的演进:**轮询/最少在途**(无状态,均衡但无缓存复用)→ **一致性哈希**(hash(前缀)定位实例,实例增减时只重映射少量 key,缓存抖动小)→ **一致性哈希 + 有界负载 CHWBL**(给每实例设负载上限 $(1+\varepsilon)\bar\rho$,热前缀超限就溢到环上下一个,亲和与防热点兼得)→ **两选一 / DualMap**(双哈希出 2 个候选实例,按当前负载挑轻的,「power of two choices」让同前缀大概率同实例、不同前缀均匀散开)。会话级亲和(按 session id)是同理的特例:把多轮对话固定在一台机,复用整段历史 KV。

![[obs-prefix亲和路由.svg]]

![[obs-112路由演进.svg]]

```python
# 一致性哈希 + 有界负载(CHWBL):亲和但防热点
import hashlib, bisect

class PrefixRouter:
    def __init__(self, replicas, eps=0.25):
        self.ring = sorted((self._h(r), r) for r in replicas)
        self.load = {r: 0 for r in replicas}
        self.eps = eps
    def _h(self, k): return int(hashlib.md5(str(k).encode()).hexdigest(), 16)
    def route(self, prefix):
        n = len(self.load); avg = sum(self.load.values()) / n
        cap = (1 + self.eps) * avg + 1          # 每实例负载上限
        idx = bisect.bisect(self.ring, (self._h(prefix),)) % n
        for j in range(n):                       # 命中上限则顺环找下一个
            r = self.ring[(idx + j) % n][1]
            if self.load[r] < cap:
                self.load[r] += 1; return r      # 同前缀→同实例,缓存复用
        return self.ring[idx][1]
```

```text
❌ 对 LLM 服务用无状态轮询(把同前缀均匀打散)
   → 每个副本都重算 2000-token 系统提示,前缀缓存命中率≈0,TTFT 高、算力浪费
✅ prefix 亲和(一致性哈希+有界负载):同前缀→同实例蹭热缓存,超限再溢出
   → 命中 RadixAttention 跳过 prefill;CHWBL 上限防止热前缀压垮单实例
```

## 面试高频
- **为什么 LLM 服务不能用普通轮询?** 轮询把同前缀打散,每个副本都重算前缀、缓存命中率≈0;LLM 的前缀缓存复用价值极大,需要亲和路由。
- **prefix affinity 为什么能提性能?** 同前缀路由到同实例 → 命中该实例的 RadixAttention 前缀缓存 → 跳过 prefill,TTFT 与算力大降。
- **亲和与负载均衡的矛盾怎么解?** 纯亲和会热点。用一致性哈希+有界负载(CHWBL)给实例设上限超限溢出,或「两选一/DualMap」双候选挑轻的,兼顾两者。
- **为何用一致性哈希而非普通 hash?** 实例增减(扩缩容/故障)时,一致性哈希只重映射少量 key,缓存抖动小;普通取模会几乎全部失效。
- **会话亲和和前缀亲和的关系?** 会话亲和按 session id 把多轮对话固定到一台机,是前缀亲和(按前缀内容)的特例,复用整段对话历史 KV。

## 关键事实
- 缓存亲和(co-locate 同前缀以复用 KV)与负载均衡天然冲突,是 LLM 路由的核心张力(**2025–2026**)。
- 会话亲和 + 前缀感知路由是 KV 缓存本地于单实例时的事实最优解(**2025**)。
- CHWBL(一致性哈希+有界负载):在保亲和的同时给每实例设负载上限,防单实例过载(**2025**,KubeAI)。
- DualMap(**2026**):双哈希映射到 2 个候选实例,按系统状态择优,用「power of two choices」同时实现亲和与均衡。
- 好的 LLM 负载均衡需同时:最大化热缓存复用、适应实例增减(少抖动)、避免单实例热点。
