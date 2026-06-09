import json
import regex
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Optional


# GPT-2 预分词正则（使用 regex 库支持 \p{L} 等 Unicode 属性）
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """Byte-level BPE Tokenizer."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ):
        """
        Args:
            vocab: {token_id: token_bytes} 映射
            merges: [(token1, token2), ...] BPE 合并规则，按创建顺序
            special_tokens: 特殊 token 列表（如 ["<|endoftext|>"]）
        """
        self.vocab = dict(vocab)  # id -> bytes
        self.merges = list(merges)

        # 处理 special tokens：如果不在 vocab 里则追加
        self.special_tokens: list[str] = special_tokens if special_tokens else []
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in set(self.vocab.values()):
                self.vocab[len(self.vocab)] = st_bytes

        # 反向映射: bytes -> id
        self.bytes_to_id: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # merge 优先级: (token1, token2) -> rank (越小越优先)
        self.merge_priority: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }

        # 编译预分词正则
        self._pat = regex.compile(GPT2_PAT)

        # 构建 special token 的正则（按长度降序匹配，保证长的优先）
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [regex.escape(t) for t in sorted_specials]
            self._special_pat = regex.compile("|".join(escaped))
        else:
            self._special_pat = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[list[str]] = None,
    ) -> "Tokenizer":
        """从文件加载 vocab 和 merges 构造 Tokenizer。"""
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        # vocab 文件格式: {token_str: id} -> 转为 {id: token_bytes}
        vocab = {int(v): k.encode("utf-8") for k, v in vocab_data.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if line and len(line.split(" ")) == 2:
                    t1, t2 = line.split(" ")
                    merges.append((t1.encode("utf-8"), t2.encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def _apply_bpe(self, token_bytes: bytes) -> list[bytes]:
        """对一个 pre-token (bytes) 应用 BPE merges，返回合并后的 pieces 列表。"""
        pieces = [bytes([b]) for b in token_bytes]

        while len(pieces) > 1:
            # 找到优先级最高（rank 最小）的相邻 pair
            best_pair = None
            best_rank = float("inf")
            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                rank = self.merge_priority.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break  # 没有可合并的 pair 了

            # 执行合并
            new_token = best_pair[0] + best_pair[1]
            new_pieces = []
            i = 0
            while i < len(pieces):
                if (
                    i < len(pieces) - 1
                    and pieces[i] == best_pair[0]
                    and pieces[i + 1] == best_pair[1]
                ):
                    new_pieces.append(new_token)
                    i += 2
                else:
                    new_pieces.append(pieces[i])
                    i += 1
            pieces = new_pieces

        return pieces

    def _encode_chunk(self, text: str) -> list[int]:
        """对一段不含 special token 的文本进行编码。"""
        ids = []
        for match in self._pat.finditer(text):
            token_bytes = match.group().encode("utf-8")
            pieces = self._apply_bpe(token_bytes)
            for piece in pieces:
                ids.append(self.bytes_to_id[piece])
        return ids

    def encode(self, text: str) -> list[int]:
        """将文本编码为 token ID 列表。"""
        if not text:
            return []

        ids: list[int] = []

        if self._special_pat is None:
            # 没有 special tokens，直接编码
            return self._encode_chunk(text)

        # 用 special token 正则切分文本
        # 找到所有 special token 的位置
        last_end = 0
        for m in self._special_pat.finditer(text):
            # 编码 special token 之前的普通文本
            start, end = m.start(), m.end()
            if start > last_end:
                ids.extend(self._encode_chunk(text[last_end:start]))
            # 编码 special token 本身
            special_bytes = m.group().encode("utf-8")
            ids.append(self.bytes_to_id[special_bytes])
            last_end = end

        # 编码最后一段普通文本
        if last_end < len(text):
            ids.extend(self._encode_chunk(text[last_end:]))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        流式编码：逐行读取，逐个 yield token ID。
        内存效率：不需要把整个文件加载到内存。
        """
        for line in iterable:
            ids = self.encode(line)
            yield from ids

    def decode(self, ids: list[int]) -> str:
        """将 token ID 列表解码为文本。"""
        byte_pieces = []
        for token_id in ids:
            byte_pieces.append(self.vocab[token_id])
        all_bytes = b"".join(byte_pieces)
        return all_bytes.decode("utf-8", errors="replace")


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    训练一个 byte-level BPE tokenizer。
    """

    # ========== Step 1: 读取文本 ==========
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ========== Step 2: 用 special tokens 切分文本（硬边界） ==========
    if special_tokens:
        escaped = [regex.escape(t) for t in special_tokens]
        split_pattern = "|".join(escaped)
        segments = regex.split(split_pattern, text)
    else:
        segments = [text]

    # ========== Step 3: 预分词，统计每个 pre-token 的出现频率 ==========
    pre_token_freqs: dict[bytes, int] = defaultdict(int)
    compiled_pat = regex.compile(GPT2_PAT)
    for segment in segments:
        for match in compiled_pat.finditer(segment):
            token_bytes = match.group().encode("utf-8")
            pre_token_freqs[token_bytes] += 1

    # ========== Step 4: 将每个 pre-token 拆成单字节序列 ==========
    word_freqs = list(pre_token_freqs.values())
    splits: list[list[bytes]] = [[bytes([b]) for b in word] for word in pre_token_freqs.keys()]

    # ========== Step 5: 计算需要做多少次 merge ==========
    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        num_merges = 0

    merges: list[tuple[bytes, bytes]] = []

    # ========== Step 6: 初始化 pair 频率和索引 ==========
    pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for wi in range(len(splits)):
        freq = word_freqs[wi]
        pieces = splits[wi]
        for i in range(len(pieces) - 1):
            pair = (pieces[i], pieces[i + 1])
            pair_freqs[pair] += freq
            pair_to_words[pair].add(wi)

    # ========== Step 7: BPE 主循环 ==========
    for _ in range(num_merges):
        if not pair_freqs:
            break

        # 选出频率最高的 pair，平局时取字典序最大的 pair
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))

        if pair_freqs[best_pair] <= 0:
            break

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]

        # 获取包含 best_pair 的所有 word
        affected_words = list(pair_to_words.pop(best_pair, set()))
        del pair_freqs[best_pair]

        for wi in affected_words:
            freq = word_freqs[wi]
            old_pieces = splits[wi]

            # 对这个 word 的旧 pieces，先移除所有旧 pair 的贡献
            for i in range(len(old_pieces) - 1):
                pair = (old_pieces[i], old_pieces[i + 1])
                if pair == best_pair:
                    continue  # 已经删除了
                pair_freqs[pair] -= freq
                if pair_freqs[pair] <= 0:
                    pair_freqs.pop(pair, None)
                pair_to_words[pair].discard(wi)
                if not pair_to_words[pair]:
                    pair_to_words.pop(pair, None)

            # 执行合并
            new_pieces = []
            i = 0
            while i < len(old_pieces):
                if (
                    i < len(old_pieces) - 1
                    and old_pieces[i] == best_pair[0]
                    and old_pieces[i + 1] == best_pair[1]
                ):
                    new_pieces.append(new_token)
                    i += 2
                else:
                    new_pieces.append(old_pieces[i])
                    i += 1
            splits[wi] = new_pieces

            # 添加新 pieces 的 pair 贡献
            for i in range(len(new_pieces) - 1):
                pair = (new_pieces[i], new_pieces[i + 1])
                pair_freqs[pair] += freq
                pair_to_words[pair].add(wi)

    # ========== Step 8: 构建最终 vocab ==========
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, (t1, t2) in enumerate(merges):
        vocab[256 + i] = t1 + t2
    for i, token in enumerate(special_tokens):
        vocab[256 + len(merges) + i] = token.encode("utf-8")

    return vocab, merges
