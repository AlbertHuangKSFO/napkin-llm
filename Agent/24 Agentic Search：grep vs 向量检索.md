[[24 Agentic Search：grep vs 向量检索|Agentic Search]] 的本质：agent 不该执着于某一种检索；它要根据问题形状、代码库规模和可用索引，在 **`rg`、AST/LSP、BM25、embedding/hybrid** 中选择候选生成器，再用可定位的代码证据验证答案。

## 直觉：问路时先选对地图

你记得「报错文本」或「配置键」时，像知道门牌号，`rg` 是最快的街道索引；你要找「这个符号到底定义在哪里、谁调用它」时，需要语法树或 LSP，像查产权登记；你只会描述「用户登录后偶尔丢会话」，词面未必匹配时，BM25/embedding 像按主题找街区。三种地图都可能把你带到附近；到达后仍要打开文件、看调用链和测试来确认。

因此「代码 agent 放弃向量检索、只用 grep」是错误的二分法。精确字面查找可先用 `rg`，语义同义词和陌生大型仓库可用 BM25/embedding/hybrid 召回，符号与调用关系应优先 AST/LSP；常见的可靠流程是**粗召回 → 语法/字面收窄 → 读取验证**。若检索通过外部 MCP 获取内容，它也属于供应链边界，见 [[AI 安全/20 Agentic 供应链与 MCP 安全]]。

## 小数字手算：BM25 为什么适合先做词面召回

设仓库有 $N=100$ 个文档，`refresh_token` 出现在 $n=9$ 个文档。BM25 的 IDF 项为：

$$
\operatorname{IDF}(q)=\ln\frac{N-n+0.5}{n+0.5}
=\ln\frac{91.5}{9.5}\approx2.27
$$

这个罕见标识符比出现在 $80$ 个文件中的 `return` 有更高区分度。若 agent 已知字面量，直接 `rg` 连建索引都不必；若用户说「刷新后登录态不见」而代码中使用的是 `rotate_cookie`，词面 BM25 可能漏掉，embedding/hybrid 就有价值。分数不是答案，打开候选文件核验才是最后一步。

## 推导：把检索看成候选集与证据链

给问题 $q$ 和仓库 $R$，检索器产生候选文件/符号集合：

$$
K=\operatorname{retrieve}(q,R,m),\qquad
E=\operatorname{verify}(K,q)
$$

`retrieve` 的优化目标偏向召回，`verify` 偏向精确。混合检索可用归一化后的词面与向量分数：

$$
\operatorname{score}(d,q)=\alpha\,\widehat{\operatorname{BM25}}(d,q)
+(1-\alpha)\,\widehat{\cos}(e_d,e_q)
$$

其中 $\alpha$ 要由仓库和 query 校准，不能把两种原始分数直接相加。之后用 `rg` 精确查标识符、用 AST/LSP 查定义/引用、用测试或构建验证因果；否则检索结果只是「看起来相关」。

![[Agentic Search：grep vs 向量检索.png]]

![[Agentic Search：grep vs 向量检索-三法对比.png]]

## 选型卡：按任务和仓库，而非信仰选工具

| 问题形状 | 首选 | 为什么 | 下一步 |
|---|---|---|---|
| 已知错误文本、API 名、配置键、正则模式 | `rg` | 无索引、可显示行号和上下文 | 打开命中处，追调用/测试 |
| 查定义、引用、重命名影响、类型层级 | AST 或 LSP | 按语法/符号而非纯字符串消歧 | LSP references 或编译器检查 |
| 自然语言 issue、术语与实现命名不同 | BM25、embedding 或 hybrid | 先召回字面不一致的候选 | 用 `rg`/AST 验证候选 |
| 大仓库、索引可维护、查询既有专名又有语义 | hybrid | 词面与语义各补盲区 | rerank 后读原始代码 |
| 小仓库、一次性排错、索引陈旧 | `rg` + AST/LSP | 建/更新向量索引成本未必回本 | 必要时再扩展检索 |

[ripgrep 官方说明](https://github.com/BurntSushi/ripgrep)表明 `rg` 默认遵守 ignore 规则并跳过隐藏/二进制文件；排查「为什么没搜到」时要知道 `--hidden`、`--no-ignore` 等开关，而不是直接断言代码不存在。LSP 的定义与引用能力来自语言服务的语义信息，[其协议概览](https://microsoft.github.io/language-server-protocol/) 可作为接口参考；AST 也要求对应语言的 parser/构建产物可用。

## 可运行代码：先用 `rg` 定位，再选择更重的检索

❌ 不限定类型和目录地扫描所有内容，或把自然语言直接当精确标识符，都会制造噪声或漏召回。

```sh
grep -R "login problem" .
```

✅ 已知标识符时，下面命令会输出行号，且只搜索 TypeScript 源码；命中后再用 LSP/AST 或阅读上下文判断语义。

```sh
rg -n --glob '*.ts' 'rotate_cookie|refresh_token' .
```

选择器本身也应把「证据类型」编码出来，而非硬编码一种技术：

```python
def choose_search(query_kind: str, repo_size: str, index_fresh: bool) -> str:
    if query_kind in {"literal", "regex", "error"}:
        return "rg"
    if query_kind in {"definition", "references", "type"}:
        return "AST/LSP"
    if query_kind == "semantic" and repo_size == "large" and index_fresh:
        return "BM25+embedding hybrid -> rg/AST verify"
    return "BM25 or rg expansion -> read verify"

assert choose_search("literal", "small", False) == "rg"
assert choose_search("references", "large", True) == "AST/LSP"
assert "hybrid" in choose_search("semantic", "large", True)
print("selection policy ok")
```

这段代码可直接运行；它不是质量模型。生产系统应从 trace 统计命中率、首个正确文件的排名、索引新鲜度、延迟与 token 消耗，再调整策略。

## Agentic loop：检索不是一次工具调用

1. 明确要验证的假设和预期证据，例如「请求入口是否把 cookie rotation 传给 session 层」。
2. 以最低成本方式生成候选：专名先 `rg`，自然语言先 BM25/embedding/hybrid，符号关系先 AST/LSP。
3. 阅读小而连续的上下文，沿调用链/数据流扩展，而不是把整个仓库塞入 [[20 上下文工程|上下文]]。
4. 用测试、类型检查、复现或第二处独立证据验证；把路径、行号和未证实假设写进 artifact。

⚠️ **向量命中不是引用。** embedding 相似度不能证明「函数调用了某 API」；同样，`rg` 命中也不能证明运行时会走到该行。代码搜索负责缩小阅读面，正确性依赖结构与执行证据。

## 面试高频
> 面试地图：[[Agent 面试题库]]

**问：什么时候优先 `rg`，什么时候上 embedding？**

答：已知标识符、错误文本、配置键或正则时优先 `rg`；自然语言描述与实现词汇不一致、代码库大且索引新鲜时，embedding/hybrid 有助于提高召回。无论哪种，都要用源码和测试验证。

**问：AST/LSP 比 grep 多了什么？**

答：它们按语言结构或语义符号识别定义、引用、类型和作用域，能区分同名变量、支持安全重命名；代价是依赖语言 server、parser 或可构建的工程环境。

**问：hybrid 为什么常见？**

答：BM25 对专名、错误码和罕见 token 强，embedding 对同义描述强；hybrid 可以扩大候选，但需要归一化、索引更新和后续 rerank/验证，不能把两种分数当作天然可比。

## 关键事实

- [ripgrep README](https://github.com/BurntSushi/ripgrep)：`rg` 是递归正则搜索工具，默认尊重 `.gitignore`/`.ignore`/`.rgignore`，并跳过隐藏和二进制文件；这解释了其快速、低噪的默认行为，也解释了某些漏搜。
- [Language Server Protocol](https://microsoft.github.io/language-server-protocol/)：提供跨编辑器的定义跳转、查找引用等语言智能接口；可用性取决于相应语言 server 与项目解析状态。
- Robertson 与 Zaragoza 的 [BM25 综述（2009）](https://dl.acm.org/doi/10.1561/1500000019) 给出词频/逆文档频率的经典概率排序框架；embedding/hybrid 是候选生成策略，仍须结合源代码证据和可运行验证。
