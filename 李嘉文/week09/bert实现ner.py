class DataGenerator(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_name"])
        self.config["vocab_size"] = len(self.tokenizer)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.label_pad_token = -1
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                if not segment.strip():
                    continue

                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentence.append(char)
                    labels.append(self.schema[label])

                self.sentences.append("".join(sentence))
                encoded = self.encode_sentence(sentence, labels)
                self.data.append(encoded)

    def encode_sentence(self, text, labels):
        # 使用BERT tokenizer处理文本
        # 注意：BERT的tokenizer可能会将一个字拆分为多个sub-token
        tokens = []
        label_ids = []

        for char, label in zip(text, labels):
            # 对每个字符进行tokenize，可能会拆分为多个sub-token
            sub_tokens = self.tokenizer.tokenize(char)
            if not sub_tokens:
                sub_tokens = ['[UNK]']

            tokens.extend(sub_tokens)
            # 第一个sub-token使用真实标签，后续的sub-token使用特殊标签（如X）
            label_ids.append(label)
            # 为后续的sub-token添加特殊标签（如果需要）
            label_ids.extend([self.schema.get('X', self.label_pad_token)] * (len(sub_tokens) - 1))

        # 添加特殊token: [CLS]和[SEP]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        label_ids = [self.label_pad_token] + label_ids + [self.label_pad_token]

        # 转换为token ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 补齐或截断
        input_ids = self.padding(input_ids)
        attention_mask = [1 if token_id != 0 else 0 for token_id in input_ids]
        label_ids = self.padding(label_ids, self.label_pad_token)

        return {
            'input_ids': torch.LongTensor(input_ids),
            'attention_mask': torch.LongTensor(attention_mask),
            'labels': torch.LongTensor(label_ids)
        }

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    # 确保Config中有bert_model_name参数
    # 例如: Config.bert_model_name = 'bert-base-chinese'
    dg = DataGenerator("../ner_data/train.txt", Config)
