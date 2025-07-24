Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数
def all_cut(sentence, Dict):
    if not sentence:
        return [[]]
    result = []
    for i in range(1, len(sentence) + 1):
        prefix = sentence[:i]
        if prefix in Dict:
            remaining = sentence[i:]
            sub_results = all_cut(remaining, Dict)
            for sub_result in sub_results:
                result.append([prefix] + sub_result)

    return result
target = all_cut(sentence, Dict)
print("切分结果：")
for i, segmentation in enumerate(target, 1):
    print(f"{i:2d}. {segmentation}")
print(f"总共找到 {len(target)} 种切分方式")

# 输出结果：
# 切分结果：
#  1. ['经', '常', '有', '意', '见', '分', '歧']
#  2. ['经', '常', '有', '意', '见', '分歧']
#  3. ['经', '常', '有', '意', '见分歧']
#  4. ['经', '常', '有', '意见', '分', '歧']
#  5. ['经', '常', '有', '意见', '分歧']
#  6. ['经', '常', '有意见', '分', '歧']
#  7. ['经', '常', '有意见', '分歧']
#  8. ['经常', '有', '意', '见', '分', '歧']
#  9. ['经常', '有', '意', '见', '分歧']
# 10. ['经常', '有', '意', '见分歧']
# 11. ['经常', '有', '意见', '分', '歧']
# 12. ['经常', '有', '意见', '分歧']
# 13. ['经常', '有意见', '分', '歧']
# 14. ['经常', '有意见', '分歧']
