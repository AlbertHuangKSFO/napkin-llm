[[12 Unbounded Consumption 成本型 DoS|Unbounded Consumption 成本型 DoS]]的核心洞察:LLM 的每次推理都直接烧钱,于是**传统 DoS 升级成"成本型 DoS"**——攻击者用极低成本的输入,撬动你极高的算力/账单,既能拖垮服务,也能单纯把你的钱包掏空。这是 OWASP LLM Top 10 的 **LLM10(Unbounded Consumption)**,把旧版的 "Denial of Service" 与 "Model Theft" 合并扩展。它向下游会变成 [[18 Agent 间通信安全与级联失败|级联放大]];与之对偶的正向优化视角见跨域 [[35 Agent 成本与延迟优化|成本优化]]。

![[sec-成本型DoS.svg]]

## 五条放大路径
- **昂贵查询**:超长上下文、强制大输出、高频并发——单条请求成本被放大几个量级。
- **上下文窗口灌爆**:把上下文塞到接近上限,推算成本随 token 数飙升。
- **递归 Agent 循环**:agent 自我调用/互相调用不收敛,无人盯着就持续烧 token(与 [[18 Agent 间通信安全与级联失败|级联失败]] 同根)。
- **钱包攻击(Denial-of-Wallet, DoW)**:目标不是让服务挂掉,而是**耗尽预算/触发天价账单**——即便服务还在跑,经济上已被击穿。
- **模型提取/窃取(Model Extraction)**:通过海量 API 查询**蒸馏复制**你的模型——窃取知识产权,同时也是一种昂贵消费;输入输出的水印/脱敏可增加难度。

## 后果
账单飙升、正常用户被拒绝服务、配额耗尽、知识产权失窃,以及在 agent 拓扑中**沿调用链级联放大**(一个失控节点拖垮一片)。

## 防御:给每条路径设硬上限
对成本型 DoS,没有"智能识别"能替代**硬上限**——核心是处处设界:
- **速率与配额**:按身份/IP 限流;**每用户预算上限**;token 配额 + 计费告警;异常用量自动熔断。
- **单次请求约束**:输入/输出 **token 截断**;请求 **timeout**;**上下文长度上限**;拒绝超大 payload。
- **Agent 与窃取专项**:**递归深度上限 + fan-out 上限**、总步数上限、循环检测(防递归烧钱);**输出脱敏防蒸馏**、查询行为异常检测、模型水印(防提取)。

## 对比表
| 路径 | 攻击者成本 | 主要损害 | 关键上限 |
|---|---|---|---|
| 昂贵查询/灌爆 | 极低 | 算力/账单 | token 截断、上下文上限、限流 |
| 钱包攻击 DoW | 低 | 预算耗尽 | 预算上限、计费熔断 |
| 递归 Agent 循环 | 低 | 失控烧钱 | 深度/fan-out/步数上限、超时 |
| 模型提取 | 中(需海量查询) | IP 失窃 | 限流、行为检测、输出水印 |

## 关键事实(含出处)
- OWASP **LLM10 "Unbounded Consumption"** 合并并扩展了旧版的拒绝服务与模型窃取,涵盖 Denial-of-Wallet、资源耗尽、模型提取等(OWASP Top 10 for LLM Applications)。
- 模型提取的可行性根植于通过 API 查询做**知识蒸馏复制模型**;输出脱敏、限流与水印是常见缓解(学界 model stealing/extraction 研究)。

## 工业界实践
成本型 DoS 工业界的第一原则:**provider 自带的限流不够用**。OpenAI/Anthropic 的速率限制是粗粒度、面向整个 API key 的,无法做"按你的终端用户/团队/模型"细分,也看不到你的业务上下文。所以企业普遍在前面架一层 **AI/LLM Gateway** 自己做配额与熔断(出于隐私/合规,也不愿把这些逻辑放在 provider 侧)。

**主流网关与防护产品**:
- **LiteLLM Proxy**(开源,事实标准):统一 100+ 模型接口,内置 per-key/per-user 预算上限、token 配额、计费告警、虚拟 key,可挂 Lakera、Presidio、Prompt Injection 等 guardrail。
- **agentgateway / Envoy AI Gateway / TrueFoundry / Azure API Management(`llm-token-limit` 策略)**:在网关层做**基于 token(而非请求数)的限流**——这是 LLM 的关键差异,因为一条 20 万 token 上下文的请求成本 ≈ 50 条 4k 请求,按请求数限流根本拦不住。
- **三层网关架构(工业界典型)**:① **token bucket** 控量(每 (user, model) 元组一个桶,空桶立即返回 429,请求不触达 provider);② **circuit breaker** 抓异常用量模式自动熔断;③ **fallback chain** 把高成本请求降级到便宜模型,保住可用性而非直接拒绝。

**纵深防御:处处设硬上限**。成本型 DoS 没有"智能识别"能替代硬上限,核心是给每条放大路径设界:
- **请求级**:输入/输出 token 截断、`max_tokens` 强制上限、请求 timeout、上下文长度上限、拒绝超大 payload。
- **身份级**:per-user 预算上限 + token 配额 + 计费告警 + 异常用量自动熔断;按身份/IP/团队分别限流(spike arrest)。
- **Agent 级**:递归深度上限、fan-out 上限、总步数上限、循环检测——防 agent 自调用/互调不收敛把账单烧穿(与级联失败同根)。
- **模型窃取专项**:见下文。

**真实案例:大规模蒸馏窃取(2025-2026)**。这是模型提取从学术走向头条的标志性事件。2025 年初 DeepSeek-R1 发布后,OpenAI 称有证据表明 DeepSeek 通过**海量查询 API + 蒸馏**复制其模型;Microsoft 安全团队报告检测到与 DeepSeek 关联账户的大规模数据提取活动。2026-02,Anthropic 进一步指控 DeepSeek、Moonshot AI、MiniMax 用**特制 prompt 海量灌入 Claude** 做"蒸馏攻击"campaign。这把 OWASP LLM10 里"模型提取"从理论变成产业级 IP 战争。
- **防蒸馏缓解(工业界)**:输出脱敏/降精度(不返回完整 logprobs/置信度)、查询行为异常检测(单账户语义覆盖面异常宽 = 蒸馏信号)、输出水印、按账户限流 + KYC、obfuscation——但 OpenAI 自己承认对手在用"新的、混淆化的方法"绕过,这是持续军备竞赛。

**误报/延迟权衡**:硬上限的代价是**误杀正常重度用户**(长上下文 RAG、批处理任务)。工业界做法:上限做成 per-tier 可配 + 突发 burst 容量 + 超限走人工审批队列而非直接 429;计费熔断设软告警(80%)+ 硬熔断(100%)两道,避免一刀切打断业务。

```python
# 简化版 token bucket:按 (user, model) 元组限流,空桶立即 429
import time
class TokenBucket:
    def __init__(self, capacity, refill_per_sec):
        self.cap, self.rate = capacity, refill_per_sec
        self.tokens, self.ts = capacity, time.time()
    def allow(self, cost):                       # cost = 该请求预估 token 数
        now = time.time()
        self.tokens = min(self.cap, self.tokens + (now - self.ts) * self.rate)
        self.ts = now
        if self.tokens >= cost:
            self.tokens -= cost; return True
        return False                             # → 网关返回 HTTP 429,不触达 provider
```

## 面试高频
**Q1:"LLM 的 DoS 和传统 Web DoS 有什么本质区别?为什么叫'成本型'?"** 传统 DoS 目标是耗尽连接/CPU 让服务挂掉;LLM 每次推理直接烧钱(GPU + token 计费),所以攻击者用**极低成本输入撬动你极高账单**——即便服务还在跑,经济上已被击穿,这就是 **Denial-of-Wallet(DoW)**。OWASP 把旧版 DoS + Model Theft 合并为 **LLM10 Unbounded Consumption**。
- 陷阱:面试官问"加 WAF / DDoS 防护够吗?"→ 不够。WAF 防流量洪峰,但一条**合法、低频、超长上下文 + 强制大输出**的请求就能烧钱,流量层面完全正常,必须在**token/成本维度**设限。

**Q2:"按请求数限流为什么对 LLM 不够?"** 因为 LLM 成本与 token 数(尤其上下文长度)强相关而非请求数:一条 20 万 token 请求 ≈ 50 条 4k 请求的成本。所以必须做 **token-based rate limiting**,并对单次请求的输入/输出 token 设硬上限。

**Q3:"模型提取/蒸馏怎么防?能彻底防住吗?"** 缓解:限流 + KYC、输出脱敏(不返回完整置信度/logprobs)、查询行为异常检测、输出水印。**不能彻底防**——只要提供 API,对手就能查询-蒸馏,防御只能提高成本和可检测性(参考 DeepSeek 事件中 OpenAI 称对手用混淆方法绕过)。这是"开放 API 可用性 vs IP 保护"的根本张力。
- 追问"agent 递归循环为什么也是成本型 DoS?"→ agent 自调用/互调不收敛会无人值守地持续烧 token,必须设递归深度/fan-out/步数上限 + 循环检测,否则单个失控节点沿调用链级联放大拖垮一片。

## 知识拓展
- **OWASP LLM Top 10:LLM10:2025 Unbounded Consumption**——合并并扩展旧版 LLM04(DoS)与 LLM10(Model Theft),涵盖 Denial-of-Wallet、资源耗尽、模型提取(genai.owasp.org)。
- **真实事件**:DeepSeek-R1 蒸馏争议(2025-01,OpenAI/Microsoft 指控);Anthropic 指控 DeepSeek/Moonshot/MiniMax 蒸馏攻击 campaign(CNBC,2026-02)。学界综述见 *A Survey on Model Extraction Attacks and Defenses for LLMs*(arXiv 2506.22521,2025)。
- **工业工具**:LiteLLM Proxy、agentgateway、Envoy AI Gateway、Azure APIM `llm-token-limit`、TrueFoundry——核心都是 token-based 限流 + per-user 预算 + 熔断。
- **框架对照**:MITRE ATLAS 有 **AML.T0024 Exfiltration via ML Inference API**(含模型提取 / `Extract ML Model`)与资源耗尽相关技术;NIST AI RMF 的 Manage 强调资源与成本风险的持续监控。
- **对抗演化**:从"暴力大请求 DoS"→"低频隐蔽 DoW(账单狙击)"→"agent 递归级联放大"→"混淆化蒸馏窃取(绕过行为检测)",防御从流量层下沉到 token/成本层与查询行为建模。

兄弟链:[[10 供应链安全(静态)|供应链安全]]、[[11 向量与嵌入弱点与 RAG 投毒|RAG 投毒]]、[[13 隐私攻击与数据保护|隐私攻击]]、[[18 Agent 间通信安全与级联失败|级联失败]]、[[24 沙箱、最小权限与人审闸门|最小权限]]。
