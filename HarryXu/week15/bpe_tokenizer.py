import json
import os


def get_stats(ids):
    # 统计相邻token对频率
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    # 合并token对
    new_ids, i = [], 0
    while i < len(ids):
        if i+1 < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPE_Tokenizer:
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}  # 初始化单字节词汇

    def train(self, folder_path):
        # 读取训练文本并转换为字节序列
        tokens = []
        for f in os.listdir(folder_path):
            if f.endswith(".txt"):
                with open(os.path.join(folder_path, f), "r", encoding="utf-8") as file:
                    tokens.extend(list(file.read().encode("utf-8")))

        print(f"训练数据总字节数: {len(tokens)}")
        ids, num_merges = list(tokens), self.vocab_size - 256

        # 执行BPE合并
        for i in range(num_merges):
            if not (stats := get_stats(ids)):  # 海象运算符简化判断
                break
            pair = max(stats, key=stats.get)
            ids = merge(ids, pair, 256 + i)
            self.merges[pair] = 256 + i
            
            if i % 100 == 0:
                print(f"合并进度: {i}/{num_merges}")

        # 构建词汇表
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        print(f"训练完成，最终词汇表大小: {len(self.vocab)}")

    def encode(self, text):
        # 文本编码为token序列
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            tokens = merge(tokens, pair, self.merges[pair])
        return tokens

    def decode(self, ids):
        # token序列解码为文本
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save_model(self, path):
        # 保存模型
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "merges": [(list(k), v) for k, v in self.merges.items()],
                "vocab": {k: list(v) for k, v in self.vocab.items()}
            }, f, ensure_ascii=False, indent=2)

    def load_model(self, path):
        # 加载模型
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data["vocab_size"]
        self.merges = {tuple(k): v for k, v in data["merges"]}
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}


def main():
    # 演示BPE分词器的使用流程
    tokenizer = BPE_Tokenizer(1024)
    print("开始训练BPE模型...")
    tokenizer.train("prts")
    tokenizer.save_model("bpe_model.json")
    print("模型已保存到 bpe_model.json")

    # 测试编码解码
    test_text = "流明是伊比利亚出身的阿戈尔人"
    print(f"\n测试文本: {test_text}")
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"编码结果: {encoded}")
    print(f"解码结果: {decoded}")
    print(f"编码解码是否正确: {test_text == decoded}")

    # 长文本测试
    with open("prts/流明.txt", "r", encoding="utf-8") as f:
        liming_text = f.read()
    
    test_long = liming_text[:100]
    print(f"\n长文本测试 (前100字):")
    print(f"原文: {test_long}")
    print(f"编码长度: {len(tokenizer.encode(test_long))} (原字节长度: {len(test_long.encode())})")
    print(f"解码一致性: {test_long == tokenizer.decode(tokenizer.encode(test_long))}")


if __name__ == "__main__":
    main()
    
