import json

input_file = "E:/BaiduNetdiskDownload/第八周 文本匹配/week8 文本匹配问题/data/valid.json"
output_file = "E:/BaiduNetdiskDownload/第八周 文本匹配/week8 文本匹配问题/data/valid_fixed.json"

with open(input_file, 'r', encoding='utf-8') as fin:
    with open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            try:
                data = json.loads(line.strip())
                if isinstance(data, list):
                    questions = data[:-1]   # 所有除了最后一个的句子作为 questions
                    target = data[-1]       # 最后一个是 target
                    json_line = json.dumps({"questions": questions, "target": target}, ensure_ascii=False)
                    fout.write(json_line + "\n")
                else:
                    print("警告：非列表行被跳过。")
            except json.JSONDecodeError:
                print("警告：解析失败的行被跳过。")
                continue
