Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

sentence = "常经有意见分歧"

def all_cut(sentence, Dict):
    results = []
    n = len(sentence)
    
    # 主回溯函数
    def backtrack(start, path, s):
        # 当处理完整个字符串
        if start == len(s):
            results.append(path[:])
            return
            
        # 1. 正常切分（不交换字符）
        for end in range(start+1, len(s)+1):
            word = s[start:end]
            if word in Dict:
                backtrack(end, path + [word], s)
        
        # 2. 特殊处理：尝试交换开头两个字符（仅当在开头时）
        if start == 0 and len(s) >= 2:
            # 创建交换后的字符串
            swapped = s[1] + s[0] + s[2:]
            # 检查交换后是否形成新词
            swapped_word = swapped[0:2]
            if swapped_word in Dict:
                # 使用交换后的字符串继续切分
                backtrack(2, path + [swapped_word], swapped)
    
    # 初始调用
    backtrack(0, [], sentence)
    
    return results

# 运行算法
results = all_cut(sentence, Dict)

# 打印所有结果
for i, cut in enumerate(results):
    print(f"{i+1}: {cut}")

# 验证结果数量
print(f"总切分数量: {len(results)}")
