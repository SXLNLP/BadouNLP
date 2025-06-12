def all_cut(sentence, dict):
    results = []

    def infunc(string, words):
        if string == "":
            results.append(words[:])
            return

        for i in range(len(string)):
            lens = i + 1
            word = string[:lens]
            if word in dict:
                new_words = words + [word]
                rest_string = string[lens:]
                infunc(rest_string, new_words)

    infunc(sentence, [])
    return results

sentence = "经常有意见分歧"
Dict = {"经常":0.1,"经":0.05,"有":0.1,"常":0.001,"有意见":0.1,"歧":0.001,"意见":0.2,"分歧":0.2,"见":0.05,"意":0.05,"见分歧":0.05,"分":0.1}
results = all_cut(sentence, Dict)
print(f"共{len(results)}种分法")
for i in results:
    print(i)