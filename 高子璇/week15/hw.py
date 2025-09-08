import os, re, heapq
from collections import Counter, defaultdict

class BPE:
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}

    def _pair_stats(self, ids):
        c = Counter(zip(ids, ids[1:]))
        return c

    def _replace_once(self, ids, pair, idx):
        pat = re.compile(r'(?<!\d)(\d+)(,\s*\d+)(?=\D|$)')
        s = ','.join(map(str, ids))
        target = '{},{}'.format(*pair)
        s = s.replace(target, str(idx))
        return list(map(int, s.split(',')))

    def train(self, text):
        num_merges = self.vocab_size - 256
        ids = list(text.encode('utf-8'))
        for i in range(num_merges):
            stats = self._pair_stats(ids)
            if not stats: break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self._replace_once(ids, pair, idx)
            self.merges[pair] = idx
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        heap = [(0, p) for p in self.merges]
        heapq.heapify(heap)
        while heap and len(ids) >= 2:
            _, (p0, p1) = heapq.heappop(heap)
            if (p0, p1) not in self.merges: continue
            ids = self._replace_once(ids, (p0, p1), self.merges[(p0, p1)])
        return ids

    def decode(self, ids):
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')

if __name__ == '__main__':
    root = r"C:\Users\e0973783\Desktop\LLM\week14 大语言模型相关第四讲\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes"
    corpus = ''.join(open(os.path.join(root, f), encoding='utf8').read() + '\n' for f in os.listdir(root))
    bpe = BPE(500)
    bpe.train(corpus)
    seq = bpe.encode('矮人直升机')
    print(seq)
    print(bpe.decode(seq))
