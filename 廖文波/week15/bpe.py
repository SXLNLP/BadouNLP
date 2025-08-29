

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 统计每个2元组出现次数
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# 合并操作:记录每个重复二元组对应的id
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



def decode(vocab, ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

def encode(merges, text):
    # 将文本转换为id序列
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        # 统计每个2元组出现次数
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break # nothing else can be merged
        idx = merges[pair]
        ids = merge(ids, pair, idx)
    return ids

def build_vocab(text, vocab_size=276):
    #转换成utf-8编码
    tokens = text.encode("utf-8")

    # ---
    # vocab_size = 276 # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
    num_merges = vocab_size - 256
    ids = list(tokens) # copy so we don't destroy the original list

    merges = {} # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx
        
    print("tokens length:", len(tokens))
    print("ids length:", len(ids))
    print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab, merges

if __name__ == "__main__":
    path = "week15 大语言模型相关第五讲/work/corpus.txt"
    corpus = load_corpus(path)
    vocab, merges = build_vocab(corpus)

    textStr = "我的家乡在日喀则，那里有条美丽的河流"
    encode_ids = encode(merges, textStr)
    text = decode(vocab, encode_ids)
    print(text)
