import re
from collections import Counter

class BPE:
    def __init__(self, vocab_size: int = 50):
        self.vocab_size = int(vocab_size)
        self.bpe_codes: dict[tuple[str, str], int] = {}
        self.idx2token: list[str] = []
        self.token2idx: dict[str, int] = {}

    def _get_stats(self, vocab: Counter) -> Counter:
        """统计相邻符号对出现的频率"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: tuple[str, str], vocab_in: dict[str, int]) -> dict[str, int]:
        vocab_out: dict[str, int] = {}
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word in vocab_in:
            w_out = p.sub("".join(pair), word)
            vocab_out[w_out] = vocab_in[word]
        return vocab_out

    def fit(self, corpus: str) -> None:
        vocab = Counter([" ".join(list(word)) + " </w>" for word in corpus.split()])

        while len(self.bpe_codes) < self.vocab_size:
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)
            self.bpe_codes[best] = len(self.bpe_codes)

        tokens = set()
        for word in vocab:
            tokens.update(word.split())
        self.idx2token = sorted(tokens)
        self.token2idx = {t: i for i, t in enumerate(self.idx2token)}

    def encode_word(self, word: str) -> list[str]:
        word_symbols: list[str] = list(word) + ["</w>"]
        while True:
            if len(word_symbols) < 2:
                break
            pairs = [(word_symbols[i], word_symbols[i + 1]) for i in range(len(word_symbols) - 1)]
            if not pairs:
                break

            pair_rank = {p: self.bpe_codes.get(p, 1e9) for p in pairs}
            if not pair_rank:
                break

            best = min(pair_rank, key=pair_rank.get)
            if pair_rank[best] == 1e9:
                break

            first, second = best
            new_word = []
            i = 0
            while i < len(word_symbols):
                if i < len(word_symbols) - 1 and word_symbols[i] == first and word_symbols[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word_symbols[i])
                    i += 1
            word_symbols = new_word
        return word_symbols

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        tokens: list[str] = []
        for word in text.split():
            tokens.extend(self.encode_word(word))
        return [self.token2idx[t] for t in tokens if t in self.token2idx]
