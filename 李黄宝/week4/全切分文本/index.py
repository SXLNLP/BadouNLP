def bulkCirculation(sentence, Dict):
    arrList, path = [], []
    def backtrack(start):
        print('  ')
        if start == len(sentence):
            arrList.append(path[:])
            return
        
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if word in Dict:
                path.append(word)
                backtrack(end)
                path.pop()
    backtrack(0)
    return arrList


Dict = {
    "经常":0.1, "经":0.05, "有":0.1, "常":0.001,
    "有意见":0.1, "歧":0.001, "意见":0.2, "分歧":0.2,
    "见":0.05, "意":0.05, "见分歧":0.05, "分":0.1
}
sentence = "经常有意见分歧"
target = bulkCirculation(sentence, Dict)
for cut in target:
    print(cut)
