"""
基于 bpe 的编码解码
1. 字符转换 utf-8 编码
2. 对编码进行两两统计合并
3. 保存合并编码
4. 基于编码后的字符进行解码
"""


def encoder(s, num=10):
    # 初始字节序列
    a = list(s.encode("utf-8"))
    print("初始字符:", s)
    print("utf-8编码:", a)

    # 编码表：字节对 -> 新编码
    vocab = {}
    code = 256  # 新编码从256开始（ASCII之后）

    # 迭代合并
    for iter in range(num):
        if len(a) < 2:
            break  # 序列长度不足，无法合并

        # 统计频率
        count = {}
        for i in range(len(a) - 1):
            pair = (a[i], a[i + 1])
            count[pair] = count.get(pair, 0) + 1

        # 如果没有字节对，则停止
        if not count:
            break

        # 找到频率最高的字节对
        max_pair = max(count, key=count.get)
        max_freq = count[max_pair]

        if max_freq == 1:
            break  # 没有频繁字节对，停止合并

        # 分配新编码
        vocab[max_pair] = code
        # print(f"迭代 {iter + 1}: 合并字节对 {max_pair} (频率{max_freq})，新编码={code}")

        # 合并字节对：创建新序列，替换所有出现max_pair的地方
        new_a = []
        i = 0
        while i < len(a):
            if i < len(a) - 1 and (a[i], a[i + 1]) == max_pair:
                new_a.append(code)
                i += 2  # 跳过两个字节
            else:
                new_a.append(a[i])
                i += 1
        a = new_a  # 更新序列
        code += 1  # 新编码递增

        print(f"合并后序列: {a}")

    print("最终序列:", a)
    print("编码表:", vocab)
    return a, vocab


def decoder(a, vocab):
    # 创建逆编码表
    rev_vocab = {code: pair for pair, code in vocab.items()}
    # 创建解码器
    decoder = []
    for i in range(len(rev_vocab)):
        tem = []
        for i in range(len(a)):
            if a[i] in rev_vocab:
                tem.extend(rev_vocab[a[i]])  # 使用 extend 展平元组
            else:
                tem.append((a[i]))  # 直接添加单个元素
        a = tem
        decoder = tem
    # 将UTF-8字节序列转换为字符串
    try:
        # 将整数列表转换为字节串，再解码为字符串
        byte_seq = bytes(a)
        decoded_str = byte_seq.decode('utf-8')
        return decoded_str,decoder
    except UnicodeDecodeError:
        # 如果解码失败，返回字节列表
        return a

if __name__ == "__main__":
    encoded_seq, vocab = encoder("北京北京北京北京", 10)
    str,decoder = decoder(encoded_seq, vocab)
    print("解码后的utf-8：",decoder)
    print("解码后的字符：", str)
