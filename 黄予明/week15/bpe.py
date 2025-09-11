#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

"""
基于DOTA2英雄介绍文件的 BPE 词表构建器

"""

def get_stats(ids):
    """
    统计相邻token对的频次
    
    Args:
        ids: token id 列表
        
    Returns:
        dict: {(id1, id2): count} 的字典
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    将指定的token对合并为新的token
    
    Args:
        ids: 原始token id列表
        pair: 要合并的token对 (id1, id2)
        idx: 新token的id
        
    Returns:
        list: 合并后的token id列表
    """
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

def load_char_vocab(vocab_file):
    """
    加载基础字符词汇表
    
    Args:
        vocab_file: 字符词表文件路径
        
    Returns:
        tuple: (char_to_id, id_to_char) 字典对
    """
    char_to_id = {}
    id_to_char = {}
    
    # 特殊token
    special_tokens = ['<UNK>', '<PAD>', '<SOS>', '<EOS>']
    for i, token in enumerate(special_tokens):
        char_to_id[token] = i
        id_to_char[i] = token
    
    # 读取字符词表
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            char = line.strip()
            if char and char not in char_to_id:
                char_id = len(char_to_id)
                char_to_id[char] = char_id
                id_to_char[char_id] = char
    
    print(f"加载基础字符词汇表: {len(char_to_id)} 个字符")
    return char_to_id, id_to_char

def load_dota2_corpus(dir_path):
    """
    加载DOTA2英雄介绍文件作为训练语料
    
    Args:
        dir_path: DOTA2英雄介绍文件目录
        
    Returns:
        str: 合并后的语料文本
    """
    print(f"开始加载DOTA2语料库: {dir_path}")
    corpus = ""
    file_count = 0
    
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    
    # 遍历目录中的所有txt文件
    for filename in os.listdir(dir_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(dir_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 清理HTML标签和特殊字符
                    content = content.replace('<br>', '\n')
                    content = content.replace('<font color="#9acd32">', '')
                    content = content.replace('<font color="#ff0000">', '')
                    content = content.replace('<font color="#87ceeb">', '')
                    content = content.replace('</font>', '')
                    content = content.replace('%', '')
                    
                    corpus += content + '\n'
                    file_count += 1
                    
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")
                continue
    
    print(f"成功加载 {file_count} 个英雄文件")
    print(f"语料库总长度: {len(corpus)} 个字符")
    
    return corpus

def text_to_ids(text, char_to_id):
    """
    将文本转换为字符ID序列
    
    Args:
        text: 输入文本
        char_to_id: 字符到ID的映射
        
    Returns:
        list: 字符ID列表
    """
    ids = []
    for char in text:
        if char in char_to_id:
            ids.append(char_to_id[char])
        else:
            ids.append(char_to_id['<UNK>'])  # 未知字符
    return ids

def build_vocab(corpus, char_to_id, id_to_char, vocab_size=1000):
    """
    构建BPE词汇表
    
    Args:
        corpus: 训练语料（字符串）
        char_to_id: 字符到ID的映射
        id_to_char: ID到字符的映射
        vocab_size: 目标词汇表大小
        
    Returns:
        tuple: (merges, vocab) 合并规则和词汇表
    """
    print(f"开始构建BPE词汇表，目标大小: {vocab_size}")
    
    # 将语料转换为字符ID序列
    ids = text_to_ids(corpus, char_to_id)
    print(f"语料包含 {len(ids)} 个字符token")
    
    # 计算需要的合并次数
    base_vocab_size = len(char_to_id)
    num_merges = vocab_size - base_vocab_size
    
    if num_merges <= 0:
        print("目标词汇表大小不大于基础词汇表，无需合并")
        return {}, id_to_char.copy()
    
    print(f"基础词汇表大小: {base_vocab_size}")
    print(f"需要进行 {num_merges} 次合并")
    
    merges = {}  # (int, int) -> int
    vocab = id_to_char.copy()  # 复制基础词汇表
    
    # 开始BPE训练
    for i in range(num_merges):
        # 统计相邻token对的频次
        stats = get_stats(ids)
        
        if not stats:
            print(f"没有更多可合并的token对，在第 {i} 次合并时停止")
            break
        
        # 选择频次最高的token对
        pair = max(stats, key=stats.get)
        new_idx = base_vocab_size + i
        
        print(f"第 {i+1} 次合并: {pair} -> {new_idx} (频次: {stats[pair]})")
        
        # 执行合并
        ids = merge(ids, pair, new_idx)
        merges[pair] = new_idx
        
        # 构建新token的字符串表示
        left_char = vocab[pair[0]]
        right_char = vocab[pair[1]]
        new_token = left_char + right_char
        vocab[new_idx] = new_token
        
        # 尝试打印可读的token（如果是有效的UTF-8字符）
        try:
            print(f"  新token {new_idx}: '{new_token}'")
        except:
            print(f"  新token {new_idx}: [不可显示]")
        
        # 每100次合并显示进度
        if (i + 1) % 100 == 0:
            print(f"已完成 {i + 1}/{num_merges} 次合并")
    
    print(f"BPE训练完成！最终词汇表大小: {len(vocab)}")
    return merges, vocab

def encode(text, merges, char_to_id):
    """
    使用BPE编码文本
    
    Args:
        text: 输入文本
        merges: BPE合并规则
        char_to_id: 字符到ID的映射
        
    Returns:
        list: 编码后的token ID列表
    """
    # 先转换为字符ID
    tokens = text_to_ids(text, char_to_id)
    
    # 应用BPE合并规则
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        # 找到优先级最高的可合并对（在merges中存在且统计频次最高）
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # 没有更多可合并的对
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    
    return tokens

def decode(ids, vocab):
    """
    解码token ID序列为文本
    
    Args:
        ids: token ID列表
        vocab: 词汇表 {id: token_string}
        
    Returns:
        str: 解码后的文本
    """
    tokens = []
    for idx in ids:
        if idx in vocab:
            tokens.append(vocab[idx])
        else:
            tokens.append('<UNK>')
    
    return ''.join(tokens)

def save_bpe_model(merges, vocab, char_to_id, save_path):
    """
    保存BPE模型（使用简单文本格式）
    
    Args:
        merges: 合并规则
        vocab: 词汇表
        char_to_id: 字符到ID映射
        save_path: 保存路径前缀
    """
    # 保存为文本格式
    txt_path = save_path + '.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        # 保存字符到ID映射
        f.write("=== CHAR_TO_ID ===\n")
        for char, char_id in char_to_id.items():
            f.write(f"{char_id}\t{char}\n")
        
        # 保存合并规则
        f.write("\n=== MERGES ===\n")
        for (id1, id2), new_id in merges.items():
            f.write(f"{id1},{id2}\t{new_id}\n")
        
        # 保存词汇表
        f.write("\n=== VOCAB ===\n")
        for token_id, token_str in vocab.items():
            f.write(f"{token_id}\t{token_str}\n")
    
    print(f"模型已保存到: {txt_path}")

def load_bpe_model(model_path):
    """
    加载BPE模型（从文本格式）
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        tuple: (merges, vocab, char_to_id)
    """
    char_to_id = {}
    merges = {}
    vocab = {}
    
    with open(model_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_section = None
    for line in lines:
        line = line.strip()
        if line == "=== CHAR_TO_ID ===":
            current_section = "char_to_id"
            continue
        elif line == "=== MERGES ===":
            current_section = "merges"
            continue
        elif line == "=== VOCAB ===":
            current_section = "vocab"
            continue
        elif not line:
            continue
        
        if current_section == "char_to_id":
            parts = line.split('\t', 1)
            char_id = int(parts[0])
            char = parts[1]
            char_to_id[char] = char_id
        elif current_section == "merges":
            parts = line.split('\t')
            pair_str = parts[0]
            new_id = int(parts[1])
            id1, id2 = map(int, pair_str.split(','))
            merges[(id1, id2)] = new_id
        elif current_section == "vocab":
            parts = line.split('\t', 1)
            token_id = int(parts[0])
            token_str = parts[1]
            vocab[token_id] = token_str
    
    print(f"模型已加载: 词汇表大小 {len(vocab)}, 合并规则 {len(merges)} 条")
    return merges, vocab, char_to_id

def main():
    """主函数 - 演示基于DOTA2语料的BPE词表构建"""
    print("=== 基于DOTA2语料的BPE词表构建器 ===\n")
    
    # 1. 加载字符词汇表
    vocab_file = "/Users/evan/Downloads/AINLP/week4 中文分词和tfidf特征应用/上午-中文分词/rnn分词/chars.txt"
    char_to_id, id_to_char = load_char_vocab(vocab_file)
    
    # 2. 加载DOTA2语料
    dota_dir = "/Users/evan/Downloads/AINLP/week14 大语言模型相关第四讲/RAG/dota2英雄介绍-byRAG/Heroes"
    training_corpus = load_dota2_corpus(dota_dir)
    
    # 3. 构建BPE词汇表
    vocab_size = 6000  # 目标词汇表大小
    merges, vocab = build_vocab(training_corpus, char_to_id, id_to_char, vocab_size)
    
    # 4. 保存模型
    save_bpe_model(merges, vocab, char_to_id, "dota2_bpe_model")
    
    # 5. 测试编码解码
    test_texts = [
        "龙骑士",
        "矮人直升机", 
        "米拉娜",
        "英雄技能",
        "攻击力",
        "魔法伤害",
        "火焰气息",
        "月神之箭",
        "追踪导弹"
    ]
    
    print(f"\n=== 编码解码测试 ===")
    for text in test_texts:
        print(f"\n原文: '{text}'")
        
        # 编码
        encoded_ids = encode(text, merges, char_to_id)
        print(f"编码结果: {encoded_ids}")
        
        # 解码
        decoded_text = decode(encoded_ids, vocab)
        print(f"解码结果: '{decoded_text}'")
        
        # 验证一致性
        if decoded_text == text:
            print("✓ 编码解码一致")
        else:
            print("✗ 编码解码不一致")
    
    # 6. 显示一些学到的合并token
    print(f"\n=== 学到的合并token示例 ===")
    merged_tokens = []
    for token_id, token_str in vocab.items():
        if token_id >= len(char_to_id) and len(token_str) > 1:  # 合并生成的token
            merged_tokens.append((token_id, token_str))
    
    if merged_tokens:
        print(f"总共学到 {len(merged_tokens)} 个合并token")
        print("示例 (按token长度排序):")
        
        # 按token长度分组显示
        by_length = {}
        for token_id, token_str in merged_tokens:
            length = len(token_str)
            if length not in by_length:
                by_length[length] = []
            by_length[length].append((token_id, token_str))
        
        # 显示不同长度的token示例
        for length in sorted(by_length.keys())[:10]:  # 显示前10种长度
            tokens = by_length[length][:5]  # 每种长度显示5个示例
            print(f"\n{length}字符token:")
            for token_id, token_str in tokens:
                print(f"  ID {token_id}: '{token_str}'")
    else:
        print("没有学到合并token")
    

if __name__ == "__main__":
    main()
