import json
import regex
from collections import defaultdict

# GPT-2 预分词正则（使用 regex 库支持 \p{L} 等 Unicode 属性）
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None) -> None:
        
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        

        
    @classmethod
    def from_file(
        cls, 
        vocab_filepath: str, 
        merge_filepath:str  , 
        special_tokens: list[str] | None = None,
        )-> "Tokenizer":

        with open(file=vocab_filepath, mode="r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        
        vocab = {int(v) : k.encode("utf-8") for k, v in vocab_data.items()}
        
        merges = []



        

    def encode():
        pass

    def encode_iterable():
        pass

    def decode():
        pass




def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]] :
    
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    

    if special_tokens:
        # 转义 special token（对包含了正则特殊字符的 | * <等）
        escaped = [regex.escape(t) for t in special_tokens]
        print(escaped)
        split_pattern = "|".join(escaped)
        segments = regex.split(split_pattern, text)
    else:
        segments = [text]
    print("segments = ", segments)

    pre_token_freqs: dict[bytes, int] = defaultdict(int)
    compiled_pat = regex.compile(GPT2_PAT)
    for segment in segments:
        for match in compiled_pat.finditer(segment):
            token = match.group().encode("utf-8")
            pre_token_freqs[token] += 1
    print(pre_token_freqs)
    
    word_freqs = list(pre_token_freqs.values())
    print(word_freqs)

    splits: list[list[bytes]] = [[bytes([b]) for b in word] for word in pre_token_freqs.keys()]

    print(splits)

    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        num_merges = 0
    
    merges: list[tuple[bytes, bytes]] = []
    
    pair_freqs: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for wi in range(len(splits)):
        freq = word_freqs[wi]
        pieces = splits[wi]
        for i in range(len(pieces) - 1):
            pair = (pieces[i], pieces[i + 1])
            pair_freqs[pair] += freq
            pair_to_words[pair].add(wi)

    print(pair_freqs)
    for _ in range(num_merges):
        if not pair_freqs:
            break 
        best_pair = max(pair_freqs, key=lambda p: (pair_freqs[p], p))
        print(best_pair)

        if(pair_freqs[best_pair] <= 0):
            break
        
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        affected_words = list(pair_to_words.pop(best_pair,set()))
        del pair_freqs[best_pair]


        # 1. after GPT2 split: list of words and its freq: word_freqs
        # 2. splits of words: bytes list: splits
        # 3. pair_to_words using pair to index which word has this pair
        # need know: 
        #   words list
        #   word split 才能重新算pair freq
        #   words_freqs
        #   pair_freqs
        #   pair_to_words 知道这个pair合并会影响哪些words
        
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
    
    
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, (t1, t2) in enumerate(merges):
        vocab[256 + i] = t1 + t2
    for i, token in enumerate(special_tokens):
        vocab[256 + len(merges) + i] = token.encode("utf-8")

    print(vocab)

train_bpe('./cs336_basics/test.txt', 258, ["<|endoftext|>"])