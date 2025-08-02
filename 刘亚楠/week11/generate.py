import torch
from transformers import BertTokenizer
from model import Title2ContentModel
from utils import load_model
from config import config


def generate(title, model, tokenizer, max_length=128):
    # 编码输入
    title_ids = tokenizer.encode(title, add_special_tokens=False)
    input_ids = (
            [tokenizer.cls_token_id] +
            title_ids +
            [tokenizer.sep_token_id] +
            [tokenizer.convert_tokens_to_ids(config.bos_token)]
    )

    # 生成内容
    for _ in range(max_length):
        # 构建attention mask
        sep_pos = len(title_ids) + 1
        attention_mask = torch.zeros((len(input_ids), len(input_ids)))
        attention_mask[:sep_pos, :sep_pos] = 1  # title部分双向
        for i in range(sep_pos, len(input_ids)):
            attention_mask[i, :i + 1] = 1  # content部分因果

        # 模型预测
        outputs = model(
            input_ids=torch.tensor([input_ids]),
            attention_mask=attention_mask.unsqueeze(0)
        )

        next_token = torch.argmax(outputs.logits[0, -1, :]).item()
        if next_token == tokenizer.convert_tokens_to_ids(config.eos_token):
            break

        input_ids.append(next_token)

    # 解码内容部分
    content_ids = input_ids[sep_pos + 1:]
    return tokenizer.decode(content_ids, skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model(config.output_dir)
    model.eval()  # 设置为评估模式
    # 示例生成
    test_titles = [
        "阿根廷歹徒抢服装尺码不对拿回店里换",
        "国际通用航空大会沈阳飞行家表演队一飞机发生坠机"
    ]

    for title in test_titles:
        content = generate(title, model, tokenizer)
        print(f"标题: {title}")
        print(f"生成内容: {content}\n")