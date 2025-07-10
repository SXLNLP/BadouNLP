# 学习率都为0.001
# 最快为Gated-CNN，时间约11ms
# 第二为LSTM，时间约36ms
# 最后为BERT，时间约874ms
class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    # def load(self):
    #     self.data = []
    #     with open(self.path, encoding="utf8") as f:
    #         for line in f:
    #             line = json.loads(line)
    #             tag = line["tag"]
    #             label = line["label"]
    #             title = line["review"]
    #             if self.config["model_type"] == "bert":
    #                 input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], truncation=True,padding='max_length')
    #             else:
    #                 input_id = self.encode_sentence(title)
    #             input_id = torch.LongTensor(input_id)
    #             label_index = torch.LongTensor([label])
    #             self.data.append([input_id, label_index])
    #     return

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            # 整体加载JSON数组
            data_list = json.load(f)  # 直接加载整个JSON数组

            for item in data_list:
                # 从对象中提取标签和评论文本
                label = int(item["label"])  # 转换为整数
                review = item["review"]

                # 使用BERT分词器编码文本
                if self.config["model_type"] == "bert":
                    # 修正：使用tokenizer()方法并处理特殊token
                    encoding = self.tokenizer(
                        review,
                        max_length=self.config["max_length"],
                        truncation=True,
                        padding='max_length',
                        return_tensors=None
                    )
                    input_id = encoding["input_ids"]
                else:
                    # 非BERT模型使用自定义编码
                    input_id = self.encode_sentence(review)

                # 转换为PyTorch张量
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])

                # 添加到数据集
                self.data.append([input_id, label_index])

        print(f"成功加载 {len(self.data)} 条评论数据")

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
