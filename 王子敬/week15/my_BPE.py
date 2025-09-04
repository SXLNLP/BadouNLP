import os
import re
import json
from collections import defaultdict, Counter


class BPE:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}

    # 获取词内部的字符对
    def get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    # 从语料库中的内容获取初始词表
    def learn_vocab(self, corpus):
        # 统计每个词的出现次数
        word_counts = Counter(corpus)

        # 初始化词表
        vocab = defaultdict(int)
        for word, count in word_counts.items():
            # 进行分词，以</w>为结束标记
            chars = list(word) + ['</w>']
            for char in chars:
                vocab[char] += count

        # 将拆分结果存入字典中
        words = {word: list(word) + ['</w>'] for word, count in word_counts.items()}
        return vocab, words, word_counts

    # 找到频率最高的字符
    def find_highest_pair(self, words, word_counts):
        pairs = defaultdict(int)
        for word, chars in words.items():
            count = word_counts[word]
            for pair in self.get_pairs(chars):
                pairs[pair] += count
        if not pairs:
            return None
        return max(pairs, key=pairs.get)

    # 构建词表
    def build_vocab(self, corpus):
        print("开始构建BPE词表。")
        vocab, words, word_counts = self.learn_vocab(corpus)

        # 初始词汇表大小
        initial_vocab_size = len(vocab)
        print(f"初始词汇表大小：{initial_vocab_size}")

        # 开始合并最高频率字符对
        iterations = self.vocab_size - initial_vocab_size
        print(f"总计要执行{iterations}次合并")
        for i in range(iterations):
            # 寻找字符对
            highest_pair = self.find_highest_pair(words, word_counts)
            if highest_pair is None:
                break

            # 合并字符对并添加到词汇表
            pair_str = "".join(highest_pair)
            vocab[pair_str] = 0
            self.merges[pair_str] = highest_pair

            # 更新词库
            new_words = {}
            for word, chars in words.items():
                new_chars = []
                j = 0
                while j < len(chars):
                    if j < len(chars) - 1 and (chars[j], chars[j + 1]) == highest_pair:
                        new_chars.append(pair_str)
                        j += 2
                    else:
                        new_chars.append(chars[j])
                        j += 1
                new_words[word] = new_chars
            words = new_words

            # 输出当前合并进度
            if (i + 1) % 500 == 0:
                print(f"已完成{i + 1}/{iterations}次合并，当前词汇表大小：{len(vocab)}")

        self.vocab = vocab
        # 生成token到ID的映射
        sorted_tokens = sorted(self.vocab.keys(), key=lambda x: (len(x), x), reverse=True)
        self.token_to_id = {token: i for i, token in enumerate(sorted_tokens)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}

        print(f"BPE词表构建完成，当前词表大小为：{len(vocab)}")

    # 保存词汇表
    def save_vocab(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for token in sorted(self.vocab.keys(), key=lambda x: len(x), reverse=True):
                f.write(f"{token}\n")
        print(f"词表已保存到{path}")

    def save_vocab_to_json(self, json_path):
        # 处理merges,将元组转换为字符串用于JSON序列化
        serializable_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}

        vocab_data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()},  # 确保键是字符串
            "merges": serializable_merges
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        print(f"已将词表保存到JSON文件中，路径为{json_path}")


# 读取文件夹中所有txt文件作为语料库
def read_corpus_from_folder(folder_path):
    corpus = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    text = text.lower()
                    # 正则表达式分割词和标点
                    words = re.findall(r'\w+|[^\w\s]', text)
                    corpus.extend(words)
                print(f"已经读取文件：{file_path}, 包含{len(words)}个词")
            except Exception as e:
                print(f"读取文件{filename}时出错：{e}")
    return corpus


if __name__ == "__main__":
    folder_path = "Heroes"
    vocab_size = 5000
    output_vocab_path = "my_BPE_vocab.txt"
    json_path = "my_BPE_vocab.json"

    print(f"正在从{folder_path}读取文件")
    corpus = read_corpus_from_folder(folder_path)
    print(f"语料库总计有{len(corpus)}个词")

    if not corpus:
        print("未找到文本，请检查文件夹路径和文件夹中的内容")
    else:
        bpe = BPE(vocab_size=vocab_size)
        bpe.build_vocab(corpus)
        # bpe.save_vocab(output_vocab_path)  # 保存到txt中
        bpe.save_vocab_to_json(json_path)

