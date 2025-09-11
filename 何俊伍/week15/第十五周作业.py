import re
from collections import defaultdict

class BPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = set()  # 存储最终词表
        self.merge_rules = []  # 存储合并规则（按顺序）

    def get_stats(self, vocab):
        """统计当前词汇中所有相邻符号对的出现频率"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab_in):
        """将词表中所有出现的字节对合并为一个新符号"""
        vocab_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in vocab_in:
            w_out = p.sub(''.join(pair), word)
            vocab_out[w_out] = vocab_in[word]
        return vocab_out

    def train(self, text):
        """训练BPE词表"""
        # 初始化词表：将文本分割为单词，添加结束符，统计频率
        words = text.split()
        vocab = defaultdict(int)
        for word in words:
            # 将单词表示为字符序列，添加结束符，用空格分隔（便于后续处理）
            processed_word = ' '.join(list(word)) + ' </w>'
            vocab[processed_word] += 1

        # 初始词表是所有唯一字符
        self.vocab = set([char for word in vocab for char in word.split()])
        
        # 迭代合并，直到达到目标词表大小
        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(vocab)
            if not pairs:
                break  # 没有更多可合并的对
            best_pair = max(pairs, key=pairs.get)
            self.merge_rules.append(best_pair)  # 记录合并规则
            vocab = self.merge_vocab(best_pair, vocab)
            # 将新合并的符号加入词表
            new_symbol = ''.join(best_pair)
            self.vocab.add(new_symbol)
        
        # 最终词表包括所有基础字符和合并得到的子词
        print(f"Final vocab size: {len(self.vocab)}")
        print("Vocab:", sorted(list(self.vocab)))

    def encode_word(self, word):
        """将一个单词编码为BPE子词序列"""
        # 初始表示：字符序列 + 结束符
        tokens = list(word) + ['</w>']
        # 应用所有合并规则（按顺序）
        for pair in self.merge_rules:
            merged = ''.join(pair)
            i = 0
            while i < len(tokens)-1:
                if tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    tokens[i] = merged
                    del tokens[i+1]
                else:
                    i += 1
        return tokens

    def encode(self, text):
        """将文本编码为BPE子词序列"""
        words = text.split()
        encoded_tokens = []
        for word in words:
            encoded_tokens.extend(self.encode_word(word))
        return encoded_tokens

# 示例用法
if __name__ == '__main__':
    # 示例文本
    text = "low lower newest widest"
    
    # 初始化并训练BPE
    bpe = BPE(vocab_size=30)
    bpe.train(text)
    
    # 编码新文本
    test_text = "lower"
    encoded = bpe.encode(test_text)
    print(f"Encoded '{test_text}':", encoded)
