import torch
import jieba
from loader import load_data, load_vocab, load_schema
from config import Config
from model import SiameseNetwork

"""
预测模块
"""


class Predictor:
    def __init__(self, config):
        self.config = config
        self.vocab = load_vocab(config["vocab_path"])
        self.schema = load_schema(config["schema_path"])  # {意图名: id}
        self.id_to_intent = {v: k for k, v in self.schema.items()}  # 反向映射
        self.model = self._load_model()
        self._precompute_knwb_vectors()

    def _load_model(self):
        model = SiameseNetwork(self.config)
        model_path = os.path.join(self.config["model_path"], "epoch_10.pth")  # 加载最后一轮模型
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model

    def _precompute_knwb_vectors(self):
        """预计算知识库向量"""
        train_data = load_data(self.config["train_data_path"], self.config, shuffle=False)
        self.knwb_vectors = []
        self.question_to_intent = []

        for intent_id, questions in train_data.dataset.knwb.items():
            for q in questions:
                with torch.no_grad():
                    vec = self.model(q.unsqueeze(0))  # 单句编码
                    self.knwb_vectors.append(vec)
                    self.question_to_intent.append(intent_id)

        self.knwb_vectors = torch.cat(self.knwb_vectors, dim=0)
        self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)

    def _encode_text(self, text):
        """文本编码为输入id"""
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        # 补齐长度
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return torch.LongTensor([input_id])

    def predict(self, text):
        """预测文本意图"""
        input_id = self._encode_text(text)
        with torch.no_grad():
            text_vec = self.model(input_id)  # 编码文本
            text_vec = torch.nn.functional.normalize(text_vec, dim=-1)

        # 计算与知识库所有向量的相似度
        similarities = torch.matmul(text_vec, self.knwb_vectors.T)
        top_idx = torch.argmax(similarities).item()

        # 映射到意图名称
        pred_intent_id = self.question_to_intent[top_idx]
        return self.id_to_intent[pred_intent_id]


if __name__ == "__main__":
    predictor = Predictor(Config)

    # 测试案例
    test_texts = [
        "改下无线套餐",
        "密码想换一下",
        "取消短信套餐",
        "查话费"
    ]

    for text in test_texts:
        intent = predictor.predict(text)
        print(f"文本: {text} → 预测意图: {intent}")