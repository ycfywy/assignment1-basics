"""
BPE 训练 Step-by-Step 标准答案展示

用一个超小的例子，展示每一步的中间状态，方便你理解和 debug。

用法:
    cd assignment1-basics
    uv run python -m cs336_basics.debug_bpe
"""

import regex
from collections import defaultdict

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def main():
    # ========== 构造一个极简 case ==========
    text = "low lower newest newest newest widest widest"
    vocab_size = 260  # 256 字节 + 4 次 merge + 0 special tokens
    special_tokens = []

    print("=" * 70)
    print("BPE 训练 Step-by-Step 标准答案")
    print("=" * 70)
    print(f"\n原始文本: {repr(text)}")
    print(f"vocab_size: {vocab_size}, special_tokens: {special_tokens}")
    print(f"需要 merge 次数: {vocab_size} - 256 - {len(special_tokens)} = {vocab_size - 256 - len(special_tokens)}")

    # ========== Step 1: 预分词 ==========
    print("\n" + "=" * 70)
    print("Step 1: 预分词 (用 GPT-2 正则切分)")
    print("=" * 70)

    compiled_pat = regex.compile(GPT2_PAT)
    pre_token_freqs: dict[bytes, int] = defaultdict(int)
    for match in compiled_pat.finditer(text):
        token_bytes = match.group().encode("utf-8")
        pre_token_freqs[token_bytes] += 1

    print(f"\n  pre-token (bytes形式)          | 频率")
    print(f"  {'─'*35}|{'─'*5}")
    for tok, freq in sorted(pre_token_freqs.items(), key=lambda x: -x[1]):
        # 同时显示可读形式
        readable = tok.decode("utf-8", errors="replace")
        print(f"  {repr(tok):35s}| {freq}    ({repr(readable)})")

    # ========== Step 2: 拆成字节序列 ==========
    print("\n" + "=" * 70)
    print("Step 2: 将每个 pre-token 拆成单字节序列")
    print("=" * 70)

    word_list = list(pre_token_freqs.keys())
    word_freqs = [pre_token_freqs[w] for w in word_list]
    splits: list[list[bytes]] = [[bytes([b]) for b in word] for word in word_list]

    print(f"\n  word (频率) => 字节序列")
    print(f"  {'─'*60}")
    for wi in range(len(word_list)):
        readable = word_list[wi].decode("utf-8", errors="replace")
        pieces_str = [p.decode("latin-1") for p in splits[wi]]
        print(f"  {repr(readable):15s} (freq={word_freqs[wi]}) => {pieces_str}")

    # ========== Step 3: 初始化 pair 频率 ==========
    print("\n" + "=" * 70)
    print("Step 3: 统计初始相邻 pair 频率 (加权)")
    print("=" * 70)

    pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for wi in range(len(splits)):
        freq = word_freqs[wi]
        pieces = splits[wi]
        for i in range(len(pieces) - 1):
            pair = (pieces[i], pieces[i + 1])
            pair_freqs[pair] += freq
            pair_to_words[pair].add(wi)

    print(f"\n  pair                         | 加权频率")
    print(f"  {'─'*35}|{'─'*10}")
    for pair, freq in sorted(pair_freqs.items(), key=lambda x: (-x[1], x[0])):
        p1 = pair[0].decode("latin-1")
        p2 = pair[1].decode("latin-1")
        print(f"  ({repr(p1):5s}, {repr(p2):5s})                | {freq}")

    # ========== Step 4: BPE 循环 ==========
    num_merges = vocab_size - 256 - len(special_tokens)
    print("\n" + "=" * 70)
    print(f"Step 4: BPE 主循环 ({num_merges} 次 merge)")
    print("=" * 70)

    merges = []
    for step in range(num_merges):
        if not pair_freqs:
            print(f"\n  [提前结束] 没有更多 pair 了")
            break

        # 选最优 pair
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
        best_freq = pair_freqs[best_pair]
        if best_freq <= 0:
            break

        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)

        p1 = best_pair[0].decode("latin-1")
        p2 = best_pair[1].decode("latin-1")
        merged = new_token.decode("latin-1")

        print(f"\n  ┌─ Merge #{step}: ({repr(p1)}, {repr(p2)}) => {repr(merged)}  [频率={best_freq}]")

        # 检查平局
        ties = [(p, f) for p, f in pair_freqs.items() if f == best_freq]
        if len(ties) > 1:
            ties_sorted = sorted(ties, key=lambda x: (-x[1], x[0]), reverse=True)
            print(f"  │  ⚠ 平局! {len(ties)} 个 pair 频率都是 {best_freq}，取字典序最大:")
            for tp, tf in ties_sorted[:3]:
                tp1 = tp[0].decode("latin-1")
                tp2 = tp[1].decode("latin-1")
                marker = " ← 选这个" if tp == best_pair else ""
                print(f"  │    ({repr(tp1)}, {repr(tp2)}){marker}")

        # 执行合并 + 增量更新
        affected_words = list(pair_to_words.pop(best_pair, set()))
        del pair_freqs[best_pair]

        for wi in affected_words:
            freq = word_freqs[wi]
            old_pieces = splits[wi]

            for i in range(len(old_pieces) - 1):
                pair = (old_pieces[i], old_pieces[i + 1])
                if pair == best_pair:
                    continue
                pair_freqs[pair] -= freq
                if pair_freqs[pair] <= 0:
                    pair_freqs.pop(pair, None)
                pair_to_words[pair].discard(wi)
                if not pair_to_words[pair]:
                    pair_to_words.pop(pair, None)

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

            for i in range(len(new_pieces) - 1):
                pair = (new_pieces[i], new_pieces[i + 1])
                pair_freqs[pair] += freq
                pair_to_words[pair].add(wi)

        # 打印合并后的所有 splits 状态
        print(f"  │")
        print(f"  │  合并后的 splits 状态:")
        for wi in range(len(word_list)):
            readable = word_list[wi].decode("utf-8", errors="replace")
            pieces_str = [p.decode("latin-1") for p in splits[wi]]
            print(f"  │    {repr(readable):15s} (freq={word_freqs[wi]}) => {pieces_str}")

        # 打印合并后的 pair 频率
        print(f"  │")
        print(f"  │  更新后的 pair 频率:")
        for pair, freq in sorted(pair_freqs.items(), key=lambda x: (-x[1], x[0])):
            pp1 = pair[0].decode("latin-1")
            pp2 = pair[1].decode("latin-1")
            print(f"  │    ({repr(pp1):8s}, {repr(pp2):8s}) => {freq}")
        print(f"  └{'─'*60}")

    # ========== Step 5: 最终 vocab ==========
    print("\n" + "=" * 70)
    print("Step 5: 构建最终 vocab")
    print("=" * 70)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, (t1, t2) in enumerate(merges):
        vocab[256 + i] = t1 + t2
    for i, token in enumerate(special_tokens):
        vocab[256 + len(merges) + i] = token.encode("utf-8")

    print(f"\n  初始 vocab: ID 0~255 = 256 个单字节")
    print(f"  Merge 产生的新 token:")
    for i, (t1, t2) in enumerate(merges):
        new_tok = t1 + t2
        print(f"    ID {256+i}: {repr(t1)} + {repr(t2)} = {repr(new_tok)} ({repr(new_tok.decode('latin-1'))})")
    if special_tokens:
        print(f"  Special tokens:")
        for i, token in enumerate(special_tokens):
            print(f"    ID {256+len(merges)+i}: {repr(token.encode('utf-8'))}")

    print(f"\n  最终 vocab 大小: {len(vocab)}")
    print(f"  merges 列表: {merges}")
    print()


if __name__ == "__main__":
    main()
