"""
    使用bpo算法的例子，课后作业的一部分
"""
import os


class MyTokenizer:
    def __init__(self, vocab_size: int):
        """
            vocab_size: 最终词表大小
        """
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        print("--开始实例化--")

    def encode(self, input_text: str, merges):
        tokens = list(input_text.encode("utf-8"))
        # merges = self.get_merges(tokens)

        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break  # nothing else can be merged
            idx = merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids, vocab):
        # vocab = {idx: bytes([idx]) for idx in range(256)}
        # merges = self.get_merges(ids)
        # for (p0, p1), idx in merges.items():
        #     vocab[idx] = vocab[p0] + vocab[p1]

        # given ids (list of integers), return Python string
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text


    def get_stats(self, ids):
        counts = {}
        # ids_after =  ids[1:]
        # list1 = zip(ids, ids_after)
        # for pair in list1:
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            # 遍历tokens，只要连续两个跟pair一致，就直接替换
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def build_vocab(self, text):
        vocab_size = self.vocab_size   # 超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
        num_merges = self.vocab_size  - 256
        tokens = text.encode("utf-8")  # raw bytes
        tokens = list(map(int, tokens))
        ids = list(tokens)

        merges = {}  # (int, int) -> int
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merging {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx

        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
            try:
                # 将unicode编码转换为可读的字符,打印出来看一看
                print(idx, vocab[idx].decode("utf8"))
            except UnicodeDecodeError:
                # 部分的词其实是部分unicode编码的组合，无法转译为完整的字符
                # 但是没关系，模型依然会把他们当成一整整体来理解
                continue

                # 实际情况中，应该把这些词表记录到文件中，就可以用于未来的的编码和解码了
        # 可以只存储merges,因为vocab总是可以通过merges再构建出来，当然也可以两个都存储
        return merges, vocab

if __name__ == "__main__":

    """   ############  以下部分可以提前生成好后保存到文件里，这样日常使用的时候就不需要每次重更新生成了  ##########  """

    dir_path = r"D:\project\xiong\BadouNLP\崔保雄\week14\old\RAG\dota2英雄介绍-byRAG\Heroes"
    # 所有文件读成一个长字符串。也可以试试只读入一个文件
    corpus = ""
    for path in os.listdir(dir_path):
        path = os.path.join(dir_path, path)
        with open(path, encoding="utf8") as f:
            text = f.read()
            corpus += text + '\n'

    tokenizer = MyTokenizer(500)

    # 构建词表
    merges, vocabs = tokenizer.build_vocab(corpus)

    """   ############  以上部分可以提前生成好后保存到文件里，这样日常使用的时候就不需要每次重更新生成了  ##########  """


    # 使用词表进行编解码
    text = "矮人直升机"
    text = "今年是中国人民抗日战争暨世界反法西斯战争胜利80周年。"
    encode_ids = tokenizer.encode(text, merges)
    print("编码结果：", encode_ids)
    decode_string = tokenizer.decode(encode_ids, vocabs)
    print("解码结果：", decode_string)
