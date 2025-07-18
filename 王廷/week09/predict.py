# predict.py
import torch
import re
from model import TorchModel
from config import Config
from transformers import BertTokenizer


def load_model(model_path):
    config = Config
    model = TorchModel(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, config


def predict(text, model, config):
    tokenizer = BertTokenizer.from_pretrained(config["bert_path"])

    # 预处理
    encoded = tokenizer.encode_plus(
        text,
        max_length=config["max_length"],
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # 预测
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    # 处理不同类型的模型输出
    if isinstance(output, list) or isinstance(output, tuple):
        # CRF模型输出是列表或元组
        pred_label = output[0]  # 取第一个样本的预测结果
    elif torch.is_tensor(output):
        # 非CRF模型输出是张量
        pred_label = torch.argmax(output, dim=-1)[0].cpu().numpy().tolist()
    else:
        raise ValueError(f"无法处理的模型输出类型: {type(output)}")

    # 解码
    entities = decode_entities(text, pred_label, attention_mask[0].tolist())
    return entities


def decode_entities(sentence, pred_label, mask):
    results = {
        "PERSON": [],
        "LOCATION": [],
        "TIME": [],
        "ORGANIZATION": []
    }

    # 确保pred_label是列表形式
    if torch.is_tensor(pred_label):
        pred_label = pred_label.cpu().numpy().tolist()

    # 提取有效标签（跳过[CLS]、[SEP]和padding）
    mask = mask if isinstance(mask, list) else mask.tolist()
    valid_indices = [i for i, m in enumerate(mask) if m == 1 and i != 0 and i != len(mask) - 1]

    # 处理不同类型的标签输入
    if isinstance(pred_label, list):
        valid_labels = [pred_label[i] for i in valid_indices]
    elif torch.is_tensor(pred_label):
        valid_labels = [pred_label[i].item() for i in valid_indices]
    else:
        raise ValueError(f"无法处理的标签类型: {type(pred_label)}")

    # 将标签序列转换为字符串模式
    label_str = ''.join(str(x) for x in valid_labels)

    # 使用正则表达式匹配实体
    # LOCATION: B-LOCATION(0) + I-LOCATION(4)
    for match in re.finditer(r"(04*)", label_str):
        s, e = match.span()
        if s < len(sentence):
            entity_text = sentence[s:e]
            results["LOCATION"].append(entity_text)

    # ORGANIZATION: B-ORGANIZATION(1) + I-ORGANIZATION(5)
    for match in re.finditer(r"(15*)", label_str):
        s, e = match.span()
        if s < len(sentence):
            entity_text = sentence[s:e]
            results["ORGANIZATION"].append(entity_text)

    # PERSON: B-PERSON(2) + I-PERSON(6)
    for match in re.finditer(r"(26*)", label_str):
        s, e = match.span()
        if s < len(sentence):
            entity_text = sentence[s:e]
            results["PERSON"].append(entity_text)

    # TIME: B-TIME(3) + I-TIME(7)
    for match in re.finditer(r"(37*)", label_str):
        s, e = match.span()
        if s < len(sentence):
            entity_text = sentence[s:e]
            results["TIME"].append(entity_text)

    return results


if __name__ == "__main__":
    # 加载模型
    model_path = "D:\\练习\\AI学习\\第九周 序列标注\\week9 序列标注问题\\week9 序列标注问题\\ner\\model_output\\epoch_10.pth"
    model, config = load_model(model_path)

    # 测试文本
    test_texts = [
        "王小明在北京大学读书，他计划2023年去纽约旅游。",
        "2025年7月18日，习近平主席在人民大会堂会见美国总统拜登。",
        "阿里巴巴集团总部位于杭州，马云是该公司的创始人。",
        "李白是唐代著名诗人，出生于碎叶城，现属吉尔吉斯斯坦。"
    ]

    for text in test_texts:
        # 预测
        entities = predict(text, model, config)

        print("\n文本:", text)
        print("识别出的实体:")
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"{entity_type}: {', '.join(entity_list)}")