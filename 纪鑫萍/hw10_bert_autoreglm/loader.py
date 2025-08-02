from transformers import BertTokenizer

"""
（1）读取数据到list
（2）创建数据集和数据加载器
    a.__init__：
        文本列表、分词器、文本最大长度、是否添加特殊标记
        预处理所有文本
    b._encode_texts：预处理所有文本
        使用分词器将所有文本编码为token ids
        提取input_ids和attention_mask并存储
    c.__len__：返回数据集的大小
    d.__getitem__：获取数据集中的单个样本
    e.create_dataset_example：创建数据集的示例
        加载预训练分词器
        构建示例文本
        创建数据集
        打印第一个样本的信息：样本0的input_ids形状、样本0的input_ids、样本0的attention_mask
        解码查看原始文本
"""


class DataGenerator:
    def __init__(self, data_path, tokenizer, max_length, with_special_tokens=True):
        self.path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_special_tokens = with_special_tokens
        self.texts = self.encode_texts(data_path)

    def encode_texts(self, data_path):
        encode_texts = []
        with open(data_path, "r", encoding="gbk") as f:
            _list_ = [line.strip() for line in f.read().split("\n") if line.strip()]
            _text_ = "".join(_list_)
            texts = [_text_[index: index + self.max_length] for index in range(0, len(_text_), self.max_length)]
            print("成功读取文本数量:", len(texts))
            for text in texts:
                # 使用分词器将所有文本编码为token ids
                encoding = self.tokenizer.encode_plus(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                    return_token_type_ids=False,
                    return_attention_mask=True,
                    add_special_tokens=self.with_special_tokens,
                )
                # 提取input_ids和attention_mask并存储（张量）
                encode_texts.append({
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                })

            return encode_texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


def create_dataset_example():
    # 加载预训练分词器
    tokenizer = BertTokenizer.from_pretrained(r"/Users/juju/BaDou/bert-base-chinese")
    # 构建示例文本
    data_path = r"/Users/juju/nlp20/class10 文本生成问题/hwAndPra/hw/corpus.txt"
    # 创建数据集
    dataset = DataGenerator(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=128
    )
    # 打印一个样本的信息：样本0的input_ids形状、样本0的input_ids、样本0的attention_mask
    sample = dataset[88]
    print(f"样本0的input_ids形状: {sample['input_ids'].shape}")
    print(f"样本0的input_ids: {sample['input_ids']}")
    print(f"样本0的attention_mask: {sample['attention_mask']}")
    # 解码查看原始文本
    decoded_text = tokenizer.decode(sample["input_ids"])
    print(f"原始文本：{decoded_text}")
    return dataset


if __name__ == "__main__":
    create_dataset_example()
