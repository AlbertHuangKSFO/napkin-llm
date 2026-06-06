[[055 分词的坑：数字、代码、多语言与 token 攻击面|分词的坑]] 讲的是子词分词在实战里反咬一口的地方:**数字被切碎导致算术变差、代码缩进与多语言被切得很费、以及分词层本身成为安全攻击面**。它接在 [[051 BPE 与 Byte-level BPE|BPE]]、[[052 WordPiece、Unigram 与 SentencePiece|WordPiece、Unigram]] 之后,告诉你「为什么 token 化不是无害的预处理」。

## 直觉

分词器是模型的「眼睛」:模型从不直接看字符,只看 token。所以**token 怎么切,直接决定模型能不能学到字符级的结构**。三个最常见的坑:

1. **数字**:BPE 只按语料频次合并,`"480"` 可能是一个 token、`"481"` 却被切成 `[4][81]`。模型看到的「同一个数」结构忽长忽短、各位数字藏在不同 token 里 → 做竖式加法时**列对不齐**,进位学不好,于是「连两个大数相加都错」。这不是 Transformer 不会加法,而是**输入被切坏了**。
2. **代码与多语言**:深缩进(8 个空格)在老分词器里几乎一空格一个 token,长代码很快撞上下文上限;一个汉字 UTF-8 占 3 字节,字节级 BPE 常把它切成 2~3 个 token(**fertility 高**),同样的窗口装的中文内容远少于英文,而且更贵。
3. **攻击面**:词表里有训练几乎没见过的**故障 token**(glitch token);Unicode 同形字 / 不可见字符能绕过字符串过滤;把特殊 token 串写进用户输入还能伪造角色。分词层成了被忽视的攻击入口,直通 [[05 Prompt Injection 提示注入|提示注入]] 与 [[06 Jailbreak 越狱|越狱]]。

一句话:**分词不是中立的预处理,它会改变模型能学什么、能装多少、以及攻击者能塞进什么。**

## 例子

**数字算术(手算对比)。** 算 `5123 + 6877 = 12000`。

理想情况(按千分位/三位一组切,各位列对齐):
```
  5 1 2 3
+ 6 8 7 7
---------
1 2 0 0 0     各位整齐 → 进位规则容易学
```

坏情况(BPE 按频次切碎,不一致):`"5123"→[51][23]`、`"6877"→[68][77]`。模型看到的是
```
  [51][23]
+ [68][77]
```
个位 `3` 和 `7` 被埋在 `[23]`、`[77]` 内部,模型**看不到"对齐的列"**,得在 token 内部脑补出各位再相加,极易错。更糟的是「差一位就换切法」:`"480"→[480]` 但 `"481"→[4][81]`,结构在数字间突变,模型无法用统一规则处理。

**补救**:GPT-4 的 `cl100k_base` 把数字固定按**三位一组、从左到右**切(`"12345"→[123][45]`),让所有 N 位数切法一致;给数字加逗号 `"1,234"` 能强制右到左对齐。Singh & Strouse(2024)实测:右到左对齐让 GPT-3.5/4 的加法准确率**提升超过 22 个百分点**。

![[tok-数字被切碎.svg]]

**代码缩进与多语言 fertility。** `"hello"` → 1 个 token;`"你好"` → 一个汉字 3 字节,常切成 2~3 个 token。同样信息量,中文/泰文等 token 数是英文的数倍。深缩进的 Python 代码在 GPT-2 里每个空格几乎独立成 token,GPT-4 学了 `"    "`(4 空格)合一,省掉一半。**fertility(平均每词切几个 token)越高,训练算力越贵、长依赖越难学**——高 fertility 可使训练成本增加约 68%。

![[tok-代码与多语言坑.svg]]

**token 攻击面(三类)。**
- **故障 token**:著名的 `SolidGoldMagikarp` 等,是词表里存在但训练语料里几乎没出现的稀有 token,其嵌入从未被训练好。触发它,GPT-3 会复读、拒答、答非所问——可被用来探测和扰乱模型。
- **Unicode 混淆**:用西里尔字母 `а`(U+0430)冒充拉丁 `a`,或插入零宽空格,绕过基于字符串匹配的关键词过滤器;但它切成的 token 与正常词不同,也就绕过了黑名单。
- **边界走私**:把 `<|endoftext|>` 或聊天模板标记写进用户输入,若编码时未禁止用户文本被识别成特殊 token,就能伪造「系统/助手」角色,构成 [[05 Prompt Injection 提示注入|提示注入]]。

![[tok-token攻击面.svg]]

## 原理

**1. 为什么数字坑源于 BPE 的频次贪心。** BPE 的合并准则只看绝对频次(见 [[051 BPE 与 Byte-level BPE|BPE]]):

$$(a,b)^\star=\arg\max_{(a,b)}\ \mathrm{count}(a,b)$$

数字串里,`"00"`、`"19"`、`"20"` 这类组合在语料(年份、价格)中高频,会被合并成块;而 `"81"`、`"47"` 不一定。于是**切法由语料统计决定,与数值结构无关** → 同一数字的不同邻居导致不同切分。要让算术稳,必须打破这种自由度:要么固定三位一组,要么在 pre-tokenization 正则里把每个数字单独拆开(`one-digit tokenization`)。

**2. fertility 的形式化。** 对语料 $C$,fertility(也叫 token-to-word ratio)

$$f=\frac{\#\text{tokens}(C)}{\#\text{words}(C)}$$

序列长度 $L\propto f$。注意力是 [[014 注意力复杂度 O(n²) 与瓶颈|$O(L^2)$]],训练/推理成本随 $f$ 上升;同样的 [[032 RoPE 外推：NTK-aware、位置插值、YaRN|上下文窗口]] 能容纳的「真实内容」随 $f$ 下降。字节级 BPE 对非拉丁脚本天然 $f$ 偏高,这是多语言模型要给小语种**足够词表配额**(见 [[059 tokenizer 训练|tokenizer 训练]])的根本原因。

**3. 故障 token 的成因。** 词表大小 $|V|$ 在 tokenizer 训练时定下,但若某些 token 主要来自被后续清洗删掉的语料(如 Reddit 用户名),它们留在词表里却几乎不出现在训练数据中。其嵌入向量 $E[i]$ 几乎没收到梯度,处于随机初始化附近 → 落到模型从未见过的表示区域 → 行为不可预测。**词表与训练语料必须用同分布数据**,否则就埋雷。

**4. 攻击面的统一视角。** 安全过滤常作用在「字符串」层,而模型作用在「token」层,两层之间存在**语义鸿沟**:同一可见文本可有多种字节/Unicode/token 表示。攻击者操纵这个鸿沟——同形字、不可见字符、特殊 token 注入——让过滤器看到的与模型看到的不一致。防御要在**规范化后**做:NFKC Unicode 规范化、剥离控制/零宽字符、编码用户输入时禁用特殊 token(`add_special_tokens` 仅限可信内容)。

**5. 数字分词三种主流策略对比。** ① **左到右三位一组**(GPT-4 `cl100k_base`):`"12345"→[123][45]`,N 位数切法一致,但加法的进位是右→左,模型仍要跨 token 对齐;② **右到左三位一组**(Llama-3、加逗号触发):`"12345"→[12][345]`,与进位方向一致,实测加法准确率显著更高;③ **逐位切**(Falcon/部分代码模型把每个数字单独成 token):`"12345"→[1][2][3][4][5]`,各位完全对齐、最稳但序列最长、fertility 高。权衡:对齐越好→算术越准,但 token 越多→越贵。Singh & Strouse(2024)实测右到左比左到右 +22 个百分点。

**6. 为什么逗号/空格能改切法(机制)。** BPE 的 pre-tokenization 正则会在标点、空格处**强制断开**,数字串被逗号分成 `1,234` 后,`234` 这种三位块从右端对齐切出,等价于强制右到左分组。这是「免训练、纯靠输入格式」改善算术的根本——没改模型,只改了喂进去的 token 序列结构。

**7. 故障 token 的数学根因(嵌入未训练)。** token $i$ 的嵌入 $E[i]$ 的更新量正比于它在训练中的出现频次。若某 token 频次 $\approx0$,$E[i]$ 几乎停留在**随机初始化**(如 $\mathcal N(0,\sigma^2)$),与正常 token 训练后形成的流形相距甚远。前向遇到它时,这个「离群」向量经过各层注意力/MLP 被放大成模型从未见过的内部状态,输出便不可预测(复读、拒答、答非所问)。补救:词表与训练语料同分布、或对长尾 token 嵌入做正则/重训。

## 代码

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")   # GPT-4 的分词器

# —— 坑1：数字切法不一致 ——
for s in ["480", "481", "1234", "12345"]:
    print(f"{s!r:>8} -> {enc.encode(s)}")
# 480   -> [19738]            一个 token
# 481   -> [19481]            也一个（cl100k 已尽量三位一组，但旧分词器会切碎）
# 12345 -> [4513, 1774]       [123][45] 三位一组、从左到右

# —— 坑2：中文 fertility 高 ——
print(len(enc.encode("hello world")))     # 2  英文很省
print(len(enc.encode("你好世界")))         # 例如 6~8  一个汉字常 2~3 token

# ❌ 错：以为给模型 "5123+6877" 它就能像人一样按位加
#    问题不在算法，在切分把各位拆散、列对不齐
# ✅ 对：推理时给数字加分隔，强制对齐
print("5,123 + 6,877")   # 逗号触发千分位/右到左对齐，算术更准
```

```python
# —— 攻击面：特殊 token 注入的防御 ——
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")

user_input = "正常文本 <|endoftext|> 伪造的系统提示"

# ❌ 危险：允许用户文本被编码成特殊 token，可伪造文档边界/角色
bad = tok(user_input)                    # 取决于配置，可能把 <|endoftext|> 识别为特殊 token

# ✅ 安全：对不可信输入禁用特殊 token，让它只当普通字符
import unicodedata
clean = unicodedata.normalize("NFKC", user_input)        # 先 Unicode 规范化
clean = "".join(c for c in clean if unicodedata.category(c)[0] != "C")  # 去控制/零宽字符
safe = tok(clean, add_special_tokens=False)              # 用户内容里不注入特殊 token
```

```python
# —— fertility 手算:同样信息,中文 token 数远多于英文 ——
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
pairs = [("hello world foo bar", 4),          # 4 个英文词
         ("你好 世界 量子 计算", 4)]            # 4 个中文词
for text, n_words in pairs:
    n_tok = len(enc.encode(text))
    print(f"{text!r:>28}  tokens={n_tok}  fertility={n_tok/n_words:.2f}")
# 英文 fertility ≈ 1.x;中文常 ≈ 2~3 —— 同窗口装的中文内容少、训练更贵
# 序列长 L ∝ fertility,注意力成本 O(L²) 随之放大,故多语言模型要给小语种更多词表配额

# —— 同形字(homoglyph)绕过黑名单的演示与防御 ——
import unicodedata
banned = "admin"
attack = "аdmin"          # 首字母是西里尔 а(U+0430),肉眼像 a
print(attack == banned)        # False —— 朴素字符串匹配漏掉
nfkc = unicodedata.normalize("NFKC", attack)
# NFKC 不会把西里尔→拉丁(不同字符),需额外的混淆字符映射(confusables)兜底
print("仍需 confusables 映射检测:", nfkc != banned)
# ✅ 防御组合:NFKC + 去零宽/控制字符 + Unicode confusables 归一 + 在规范化后再过滤
```

## 面试高频

- **Q:为什么 LLM 连大数加法都会错?** A:多半不是不会算,而是**分词把数字切碎且不一致**,各位数字藏在不同 token 内部、列对不齐,进位规则学不好。补救:固定三位一组切、给数字加逗号强制右到左对齐(实测加法准确率 +22 个百分点)、或逐位切。
- **Q:什么是 fertility,为什么对多语言重要?** A:平均每个词切成几个 token。字节级 BPE 对中文/泰文等非拉丁脚本 fertility 高(一个汉字 2~3 token),导致序列更长、算力更贵($O(L^2)$)、上下文装得更少、长依赖更难学;多语言模型要给小语种足够词表配额来压低它。
- **Q:代码缩进为什么是坑?** A:深缩进的空格在老分词器里几乎一空格一 token,长代码迅速撞上下文上限;现代分词器学了多空格合并的 token(如 4 空格合一)来缓解。
- **Q:什么是故障 token(glitch token)?** A:词表里存在但训练几乎没见过的稀有 token,嵌入未被训练好,触发会让模型行为异常(复读、拒答)。根因:词表与训练语料分布不一致(如保留了被清洗删掉的用户名 token)。
- **Q:分词层有哪些安全风险?** A:① 故障 token 扰动;② Unicode 同形字/不可见字符绕过字符串过滤([[05 Prompt Injection 提示注入|提示注入]]/[[06 Jailbreak 越狱|越狱]]);③ 把特殊 token 写进用户输入伪造角色。防御:NFKC 规范化 + 去隐藏字符 + 对不可信输入禁用特殊 token。
- **Q:数字分词有哪几种策略,各自权衡?** A:左到右三位一组(切法一致但与进位反向)、右到左三位一组(与进位同向,加法更准,加逗号可触发)、逐位切(各位完全对齐最稳但 fertility 最高)。对齐越好算术越准,token 越多越贵。
- **Q:为什么给数字加逗号能提升算术?** A:逗号在 pre-tokenization 处强制断开,把数字按三位从右端对齐切出,等价于右到左分组、与进位方向一致;纯靠输入格式、不改模型(免训练)。
- **Q:同形字攻击 NFKC 能挡住吗?** A:挡不全。NFKC 做兼容性规范化(全角→半角、连字等),但西里尔 `а`→拉丁 `a` 属不同字符、不在 NFKC 范围,需额外的 Unicode confusables(易混淆字符)映射兜底,再去零宽/控制字符。
- **陷阱**:字符串层过滤 ≠ token 层安全;同一可见文本有多种 token 表示,过滤要在规范化之后做;NFKC 不覆盖同形字,需 confusables 补充;数字对齐策略选错会让算术系统性偏弱。

## 关键事实

- 数字分词影响算术:**Singh & Strouse《Tokenization counts》(arXiv:2402.14903, 2024;ICLR 2025)**——右到左(R2L)分词比左到右使 GPT-3.5/GPT-4 加法准确率提升超过 22 个百分点;加逗号可强制 R2L 对齐。
- GPT-4 `cl100k_base` 将数字固定按**最多三位一组**切分(相比 GPT-2/3 更一致),缓解但未根除数字坑。
- tokenizer 选择对多语言影响显著、fertility 决定算力:**《Tokenizer Choice For LLM Training》(arXiv:2310.08754, NAACL Findings 2024)**——高 fertility 可使训练算力增加约 68%,并损害长依赖学习;英文 ~33k 词表足够,多语言需数倍。
- 故障 token:`SolidGoldMagikarp` 等,见 Rumbelow & Watkins(2023, LessWrong);成因是词表含训练几乎未见的稀有 token。
- 字节级 BPE 零 OOV 但非拉丁脚本 fertility 高(一个汉字 UTF-8 = 3 字节),见 [[051 BPE 与 Byte-level BPE|Byte-level BPE]]。
- 关联:子词三流派 [[051 BPE 与 Byte-level BPE|BPE]] / [[052 WordPiece、Unigram 与 SentencePiece|WordPiece、Unigram]];词表与特殊 token [[053 词表、特殊 token 与对话模板|词表]];分词器训练与词表大小 [[059 tokenizer 训练|tokenizer 训练]];安全侧 [[05 Prompt Injection 提示注入|提示注入]]、[[06 Jailbreak 越狱|越狱]]。
