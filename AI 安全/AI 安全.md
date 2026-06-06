# AI 安全学习地图

> 当模型会自己调 API、写记忆、连工具,**自然语言就成了新的攻击载体**,传统 AppSec 不再够用。这一域按 2026 已成型的**三层栈**展开:**模型层**(OWASP LLM Top 10 v2)管「模型怎么被操纵」、**Agent 层**(OWASP Top 10 for Agentic Applications / ASI)管「操纵被赋予自主性后会发生什么——被投毒的指令变成一次 API 调用」、**协议层**(OWASP MCP Top 10)管工具与连接。攻击沿栈**级联**,所以先建威胁全景与 safety≠security 的边界,再逐层拆攻击面(注入/越狱/投毒/泄露/过度代理/身份/记忆/MCP),然后落到**防御与运营**(护栏、红队、评估基准、沙箱、监控)与**框架治理**(OWASP/MITRE ATLAS/NIST/EU AI Act/ISO 42001 如何并用)。Agent 层风险的根在 [[02 Workflow 与 Agent 的边界|Agent 自主性]],检索侧投毒接 RAG 域的 [[16 检索安全与访问控制|检索安全]]。这里只放定位与链接,不写正文。

## ① 基础
- [[01 AI 安全总览与三层栈|AI 安全总览与三层栈]]
- [[02 LLM 攻击面与威胁建模|LLM 攻击面与威胁建模]]
- [[03 内容安全与对齐边界|内容安全与对齐边界]]
- [[04 AI 赋能的攻击(Offensive AI)|AI 赋能的攻击(Offensive AI)]]

## ② 模型层 · OWASP LLM Top 10
- [[05 Prompt Injection 提示注入|Prompt Injection 提示注入]]
- [[06 Jailbreak 越狱|Jailbreak 越狱]]
- [[07 敏感信息泄露与 System Prompt 泄露|敏感信息泄露与 System Prompt 泄露]]
- [[08 不安全输出处理|不安全输出处理]]
- [[09 数据与模型投毒|数据与模型投毒]]
- [[10 供应链安全(静态)|供应链安全(静态)]]
- [[11 向量与嵌入弱点与 RAG 投毒|向量与嵌入弱点与 RAG 投毒]]
- [[12 Unbounded Consumption 成本型 DoS|Unbounded Consumption 成本型 DoS]]
- [[13 隐私攻击与数据保护|隐私攻击与数据保护]]

## ③ Agent 层 · OWASP ASI Top 10
- [[14 Excessive Agency 与 Goal Hijack|Excessive Agency 与 Goal Hijack]]
- [[15 Tool Misuse 与意外代码执行|Tool Misuse 与意外代码执行]]
- [[16 Agent 身份与权限滥用(非人类身份 NHI)|Agent 身份与权限滥用(NHI)]]
- [[17 Memory 与 Context Poisoning|Memory 与 Context Poisoning]]
- [[18 Agent 间通信安全与级联失败|Agent 间通信安全与级联失败]]
- [[19 人机信任利用与 Rogue Agents|人机信任利用与 Rogue Agents]]
- [[20 Agentic 供应链与 MCP 安全|Agentic 供应链与 MCP 安全]]

## ④ 防御与运营
- [[21 Guardrails 与输入输出防护|Guardrails 与输入输出防护]]
- [[22 AI 红队与对抗测试|AI 红队与对抗测试]]
- [[23 安全评估基准|安全评估基准]]
- [[24 沙箱、最小权限与人审闸门|沙箱、最小权限与人审闸门]]
- [[25 监控、可观测与事件响应|监控、可观测与事件响应]]

## ⑤ 框架与治理
- [[26 安全框架与治理地图|安全框架与治理地图]]

## ⑥ 跨域桥接 · 安全是横切关注点
- [[02 Workflow 与 Agent 的边界|Workflow 与 Agent 的边界]](「最小自由度」本质是安全控制:Agent 层攻击的根因)
- [[23 Agent Harness 概览|Agent Harness 概览]](Safety 是 harness 四职责之一,沙箱/审批落在这层)
- [[16 工具设计与工具层|工具设计与工具层]] · [[17 MCP 模型上下文协议|MCP 模型上下文协议]](工具与协议层的攻击面源头)
- [[16 检索安全与访问控制|RAG 检索安全与访问控制]] · [[17 检索数据治理|RAG 检索数据治理]](检索侧投毒与权限,与 ② 模型层 RAG 投毒互链)
- [[38 Agent 评估与可观测性|Agent 评估与可观测性]](运行时监控与安全可观测同源)
