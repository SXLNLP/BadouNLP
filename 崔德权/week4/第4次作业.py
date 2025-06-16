
import re
import time

# 加载字典
def get_work_dict(Dict):
    max_word_length = 0
    word_set = set()
    for word in Dict.keys():
        word_set.add(word)
        max_word_length = max(max_word_length, len(word))
    return word_set, max_word_length

def all_cut(sentence):
    if not sentence:
        return []
    
    # 获取词典中最长词的长度
    word_set, max_len = get_work_dict(Dict)
    # 存储所有切分结果
    results = []  
    
    # DFS递归函数
    def dfs(start, split_list):
        if start == len(sentence):
            results.append(split_list[:])
            return
        
        # 尝试从start开始
        for end in range(start + 1, min(start + max_len + 1, len(sentence) + 1)):
            word = sentence[start:end]
            if word in word_set:
                # 找到匹配词，继续向后切分
                split_list.append(word)
                dfs(end, split_list)
                split_list.pop()
    
    # 从起始位置开始DFS
    dfs(0, [])
    return results

# 词典
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

#待切分文本
sentence = "经常有意见分歧"
print(all_cut(sentence))
