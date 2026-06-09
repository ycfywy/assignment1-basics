# BPE Tokenizer — Python API & 语法笔记

## 1. `regex.split(pattern, text)` — 按分隔符切分文本

```python
# 用 special tokens 作为分隔符切开文本
escaped = [regex.escape(t) for t in special_tokens]  # 转义 | * < 等正则特殊字符
split_pattern = "|".join(escaped)                     # 用 | (或) 拼接: "tok1|tok2|tok3"
segments = regex.split(split_pattern, text)           # 切开，分隔符本身被丢弃
```

- `regex.escape(t)`: 把字符串里的正则特殊字符前加 `\`，让它们当普通文本匹配
- split 后返回的是 `list[str]`，分隔符不保留
- 传字符串或编译好的 pattern 都可以；只调用一次的话直接传字符串即可

---

## 2. `defaultdict` — 带默认值的字典

```python
from collections import defaultdict

# 普通 dict 访问不存在的 key → KeyError
# defaultdict 访问不存在的 key → 自动调用工厂函数创建默认值
d = defaultdict(int)    # int() → 0
d = defaultdict(list)   # list() → []
d = defaultdict(set)    # set() → set()
```

常见用法模式：
```python
freqs = defaultdict(int)
freqs[key] += 1              # 无需 if key not in: freqs[key] = 0

pair_to_words = defaultdict(set)
pair_to_words[pair].add(wi)  # 无需手动创建空 set
```

---

## 3. `finditer` — 返回类型 & 处理方式

```python
# finditer 返回 Iterator[Match]，惰性迭代
for m in pat.finditer(text):
    m.group()   # str  — 匹配到的完整文本
    m.start()   # int  — 匹配起始位置
    m.end()     # int  — 匹配结束位置
    m.span()    # (int, int) — (start, end)
```

| 方法 | 返回 | 特点 |
|---|---|---|
| `finditer` | `Iterator[Match]` | 惰性，有位置信息 |
| `findall` | `list[str]` | 一次性，无位置 |

编译一次 vs 传字符串：多次复用时先 `compile` 更高效，只用一次直接传字符串即可。

---

## 4. `pair_to_words` — 反向索引

```python
pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
#                 key: pair (如 (b'e',b'r'))    value: 包含这个 pair 的 word 索引集合
```

作用：快速找到"哪些 word 里包含这个 pair"，BPE 合并时只需更新受影响的 word，避免全量扫描。

初始化时构建：
```python
for wi in range(len(splits)):
    for i in range(len(pieces) - 1):
        pair = (pieces[i], pieces[i + 1])
        pair_to_words[pair].add(wi)   # defaultdict(set) 自动创建空 set
```

---

## 5. `max(iterable, key=函数)` — 按自定义规则找最大值

```python
# key 是一个函数，作用于每个元素，按返回值比较，但返回原始元素
max(pair_freqs, key=lambda p: pair_freqs[p])     # 按频率取最大
max(pair_freqs, key=lambda p: (pair_freqs[p], p)) # 频率优先，平局按 pair 字典序
```

- `lambda p: expr`: 匿名函数，`p` 是参数名（可任意取）
- 返回元组时按字典序比较：先比第一个，平局比第二个

---

## 6. `dict.pop(key, default)` — 读 + 删 + 安全

```python
d.pop(key)            # 读取并删除；key 不存在 → KeyError
d.pop(key, default)   # 读取并删除；key 不存在 → 返回 default，不报错
d[key]                # 只读不删；key 不存在 → KeyError
```

为什么 BPE 中用 `.pop` 而不用 `[]`：
- 合并 `(e,r)` → `er` 后，旧 pair `(e,r)` 不再存在，需要从索引中删除
- `.pop()` 一行完成"读出 + 删除"，比 `[]` + `del` 更简洁
- 传 `set()` 作为默认值是防御性写法：万一数据不一致也不会崩溃
