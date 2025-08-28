import re
from collections import defaultdict, Counter


class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def train(self, corpus):
        # 初始化
        word_freq = Counter()
        for text in corpus:
            words = text.split()
            for word in words:
                # 添加</w>作为文件边界
                word_freq[' '.join(list(word)) + ' </w>'] += 1

        # 合并
        while len(word_freq) < self.vocab_size:
            pairs = defaultdict(int)
            for word, freq in word_freq.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)

            new_word_freq = {}
            bigram = re.escape(' '.join(best_pair))
            pattern = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

            for word in word_freq:
                new_word = pattern.sub(''.join(best_pair), word)
                new_word_freq[new_word] = word_freq[word]

            word_freq = new_word_freq

        # 词表汇总
        self.vocab = set()
        for word in word_freq:
            for token in word.split():
                self.vocab.add(token)
        self.vocab = sorted(self.vocab)

    def encode(self, text):
        tokens = []
        for word in text.split():
            word = list(word) + ['</w>']
            while len(word) > 1:
                pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
                best_pair = min(
                    pairs,
                    key=lambda pair: self.merges.index(pair) if pair in self.merges else float('inf')
                )

                if best_pair not in self.merges:
                    break
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word
            tokens.extend(word)
        return tokens


if __name__ == "__main__":
    corpus = [
        "英雄名：德鲁伊",
        "背景故事 德鲁伊巨熊部落 天生智者 专注探索 理解自然秩序",
        "熊灵伙伴 灵魂链接",
    ]

    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(corpus)

    text = "德鲁伊 背景故事 灵魂链接"
    print("词表:", tokenizer.vocab)
    print(f"结果 '{text}':", tokenizer.encode(text))
    '''
    结果：
    词表: ['专注探索</w>', '天生智者</w>', '德鲁伊巨熊部落</w>', '灵魂链接</w>', '熊灵伙伴</w>', '理解自然秩序</w>', '背景故事</w>', '英雄名：德鲁伊</w>']
结果 '德鲁伊 背景故事 灵魂链接': ['德鲁伊', '</w>', '背景故事</w>', '灵魂链接</w>']
    '''