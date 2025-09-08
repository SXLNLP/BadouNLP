import os


def get_stats(ids):
    """统计相邻字节对的出现频率"""
    counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """合并指定的字节对"""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def train_bpe(text, vocab_size=500):
    """训练BPE词表"""
    # 初始化为UTF-8字节
    ids = list(text.encode("utf-8"))
    merges = {}  # 存储合并规则

    # 进行多次合并
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        if not stats:  # 没有可合并的字节对
            break

        # 找到最频繁的字节对
        pair = max(stats, key=stats.get)
        new_id = 256 + i

        # 执行合并
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id

        # 打印合并信息
        try:
            # 尝试构建当前词汇表并解码
            vocab = {idx: bytes([idx]) for idx in range(256)}
            for (p0, p1), idx in merges.items():
                vocab[idx] = vocab[p0] + vocab[p1]
            print(f"合并 {pair} -> {new_id} ({vocab[new_id].decode('utf8')})")
        except:
            print(f"合并 {pair} -> {new_id}")

    # 构建最终词汇表
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]

    return merges, vocab


def encode(text, merges):
    """编码文本为BPE标记"""
    tokens = list(text.encode("utf-8"))

    # 应用所有合并规则
    for pair, new_id in merges.items():
        tokens = merge(tokens, pair, new_id)

    return tokens


def decode(ids, vocab):
    """解码BPE标记为文本"""
    # 将标记转换为字节
    byte_string = b"".join(vocab[idx] for idx in ids)
    # 解码为文本
    return byte_string.decode("utf-8", errors="replace")


if __name__ == "__main__":
    # 读取语料库
    corpus = ""
    dir_path = r"D:\learn\python39\pythonProject250704\week15\dota2英雄介绍-byRAG\Heroes"
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                corpus += f.read() + "\n"

    # 训练BPE
    merges, vocab = train_bpe(corpus, vocab_size=500)

    # 测试编码解码
    test_text = "冥魂大帝"
    encoded = encode(test_text, merges)
    decoded = decode(encoded, vocab)

    print(f"原始文本: {test_text}")
    print(f"编码结果: {encoded}")
    print(f"解码结果: {decoded}")
