
from collections import deque
arrlist = ["经常", "经", "有", "常", "有意见", "歧", "意见", "分歧", "见", "意", "见分歧", "分"]
sentence = "经常有意见分歧"
def all_cut(sentence, Dict):
    word_set = set(Dict)
    result = []
    queue = deque([(0, [])])
    while queue:
        current_pos, path = queue.popleft()
        if current_pos == len(sentence):
            result.append(path)
            continue
        for end in range(current_pos + 1, len(sentence) + 1):
            word = sentence[current_pos:end]
            if word in word_set:
                new_path = path + [word]

                queue.append((end, new_path))
    
    return result
