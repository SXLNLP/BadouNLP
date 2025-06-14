
import re
import time
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

def all_cut(sentence, Dict):
    result = []
    stack = [(0, [])]  # (start_index, current_partition)
    start_time = time.time()
    while stack:
        start, path = stack.pop()
        if start == len(sentence):
            result.append(path)
            continue
        # 尝试所有可能的结束位置
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                new_path = path + [word]
                stack.append((end, new_path))
    print("耗时：", time.time() - start_time)
    return result
sentence = "经常有意见分歧"
#print(all_cut(sentence, Dict))
a = all_cut(sentence, Dict)
for i in a:
    print(i)
