from train_text import train_text
# BPE Tokenizer

class BPETokenizer:
    def __init__(self, text, merges=None, vocab=None):
        self.merges = merges or {}
        self.vocab = vocab or {}
        self.build_vocab(text)

    def build_vocab(self, text, vocab_size=300):
        # ... build vocab logic ...
        tokens = text.encode("utf-8")  # raw bytes
        ids = list(tokens)  # convert to a list of integers in range 0..255 for convenience

        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # print(f"merging {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx

        ori_vocab = {idx: bytes([idx]) for idx in range(256)}
        self.vocab.update(ori_vocab)
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, text):
        # ... encoding logic ...
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)

        return tokens


    def decode(self, ids):
        # ... decoding logic ...
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text


if __name__ == '__main__':
    bpe = BPETokenizer(train_text)

    test_text = "hello world!"
    encoded = bpe.encode(test_text)
    decoded = bpe.decode(encoded)

    print(f"原始文本: {test_text}")
    print(f"编码结果: {encoded}")
    print(f"解码结果: {decoded}")
    print(f"编码解码是否一致: {test_text == decoded}")
