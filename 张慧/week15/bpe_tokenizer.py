# -*- encoding: utf-8 -*-
"""
@File    : bpe_tokenizer.py
@Time    : 2025/9/10 17:22
@Version : python 3.10
@Author  : Hui Zhang
@Contact : hui.zhang@quanmag.com
@Software: PyCharm
@Description:
"""


class BPETokenizer:
    def __init__(self, vocab=None, merges=None):
        """初始化BPE分词器

        Args:
            vocab (dict): 词汇表，将token ID映射到字节序列
            merges (dict): 合并规则，将token对映射到新的token ID
        """
        self.vocab = vocab or {idx: bytes([idx]) for idx in range(256)}
        self.merges = merges or {}

    def train(self, text, vocab_size):
        """训练BPE模型

        Args:
            text (str): 用于训练的文本
            vocab_size (int): 期望的词汇表大小
        """
        # 将文本转换为UTF-8字节，然后转换为整数列表
        tokens = list(text.encode("utf-8"))

        # 计算需要合并的次数
        num_merges = vocab_size - 256
        ids = list(tokens)  # 复制一份，避免修改原始列表

        # 重置merges和vocab
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        # 执行BPE训练
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self._merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        """将文本编码为token ID列表

        Args:
            text (str): 待编码的文本

        Returns:
            list: token ID列表
        """
        if not isinstance(text, str):
            raise TypeError("输入必须是字符串")

        # 处理空字符串情况
        if len(text) == 0:
            return []

        # 将文本转换为UTF-8字节，然后转换为整数列表
        tokens = list(text.encode("utf-8"))

        # 重复合并最常见的相邻token对
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            if not stats:
                break

            # 找到最常见的token对
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # 如果这个token对不在merges中，说明无法继续合并
            if pair not in self.merges:
                break

            # 合并token对
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)

        return tokens

    def decode(self, ids):
        """将token ID列表解码为文本

        Args:
            ids (list): token ID列表

        Returns:
            str: 解码后的文本
        """
        # 将token ID转换为字节序列
        tokens = b"".join(self.vocab[idx] for idx in ids)

        # 将字节序列解码为字符串
        text = tokens.decode("utf-8", errors="replace")
        return text

    def _get_stats(self, ids):
        """统计相邻token对的频率

        Args:
            ids (list): token ID列表

        Returns:
            dict: token对到频率的映射
        """
        counts = {}
        for pair in zip(ids, ids[1:]):  # Pythonic way to iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        """合并指定的token对

        Args:
            ids (list): token ID列表
            pair (tuple): 要合并的token对
            idx (int): 合并后的新token ID

        Returns:
            list: 合并后的token ID列表
        """
        newids = []
        i = 0
        while i < len(ids):
            # 如果找到要合并的token对
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def save(self, file_prefix):
        """保存模型到文件

        Args:
            file_prefix (str): 文件前缀
        """
        import json

        # 保存词汇表
        vocab_file = file_prefix + '.vocab'
        with open(vocab_file, 'w', encoding='utf-8') as f:
            # 将bytes转换为可序列化的格式
            serializable_vocab = {k: v.decode('utf-8', errors='replace') if isinstance(v, bytes) else v
                                  for k, v in self.vocab.items()}
            json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)

        # 保存合并规则
        merges_file = file_prefix + '.merges'
        with open(merges_file, 'w', encoding='utf-8') as f:
            # 将tuple键转换为字符串
            serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
            json.dump(serializable_merges, f, ensure_ascii=False, indent=2)

    def load(self, file_prefix):
        """从文件加载模型

        Args:
            file_prefix (str): 文件前缀
        """
        import json

        # 加载词汇表
        vocab_file = file_prefix + '.vocab'
        with open(vocab_file, 'r', encoding='utf-8') as f:
            loaded_vocab = json.load(f)
            # 将字符串转换回bytes
            self.vocab = {int(k): v.encode('utf-8') if isinstance(v, str) else v
                          for k, v in loaded_vocab.items()}

        # 加载合并规则
        merges_file = file_prefix + '.merges'
        with open(merges_file, 'r', encoding='utf-8') as f:
            loaded_merges = json.load(f)
            # 将字符串键转换回tuple
            self.merges = {tuple(map(int, k.split(','))): v for k, v in loaded_merges.items()}


# 使用示例
if __name__ == "__main__":
    # 创建分词器实例
    tokenizer = BPETokenizer()

    # 训练文本
    text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."

    # 训练模型
    tokenizer.train(text, 300)

    # 测试编码和解码
    test_text = "A Programmer’s Introduction to Unicode"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"原始文本: {test_text}")
    print(f"编码结果: {encoded}")
    print(f"解码结果: {decoded}")
    print(f"编码解码是否一致: {test_text == decoded}")

    # 保存模型
    tokenizer.save("bpe_model")

    # 加载模型
    new_tokenizer = BPETokenizer()
    new_tokenizer.load("bpe_model")

    # 验证加载的模型
    encoded2 = new_tokenizer.encode(test_text)
    print(f"加载模型后的编码结果: {encoded2}")
    print(f"两次编码结果是否一致: {encoded == encoded2}")
