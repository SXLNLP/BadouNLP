from transformers import BertModel


def bertParameterCount():
    bert = BertModel.from_pretrained(r"F:\八斗ai课程\06-第六周 语言模型\bert-base-chinese", return_dict=False)

    state_dict = bert.state_dict()

    # 总参数量
    # sum_parameters = sum(p.numel() for p in bert.parameters())
    sum_parameters = sum(p.numel() for p in bert.state_dict().values())
    print("总参数量：{}个".format(sum_parameters))

    # 可训练参数量
    sum_practice_parameters = sum(p.numel() for p in bert.parameters() if p.requires_grad)
    print("可训练参数量：{}个".format(sum_practice_parameters))


bertParameterCount()
