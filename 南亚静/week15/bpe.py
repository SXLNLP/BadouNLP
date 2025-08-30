import os

'''
bpe构建词表示范
'''

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def build_vocab(text):
    vocab_size = 500 # 超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
    num_merges = vocab_size - 256
    tokens = text.encode("utf-8") # raw bytes
    tokens = list(map(int, tokens))
    ids = list(tokens) 

    merges = {} # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
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

    #实际情况中，应该把这些词表记录到文件中，就可以用于未来的的编码和解码了
    #可以只存储merges,因为vocab总是可以通过merges再构建出来，当然也可以两个都存储
    return merges, vocab

#编码过程
def decode(ids, vocab):
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text

#解码过程
def encode(text, merges):
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


if __name__ == "__main__":
    dir_path = r"E:\badou\八斗课程\week14 大语言模型相关第四讲\RAG\dota2英雄介绍-byRAG\Heroes"
    #所有文件读成一个长字符串。也可以试试只读入一个文件
    corpus = ""
    for path in os.listdir(dir_path):
        path = os.path.join(dir_path, path)
        with open(path, encoding="utf8") as f:
            text = f.read()
            corpus += text + '\n'
    #构建词表
    merges, vocabs = build_vocab(corpus)
    #使用词表进行编解码
    string = "矮人直升机"
    encode_ids = encode(string, merges)
    print("编码结果：", encode_ids)
    decode_string = decode(encode_ids, vocabs)
    print("解码结果：", decode_string)
