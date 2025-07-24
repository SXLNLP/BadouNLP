import torch
import os
from loader import load_data

"""
模型评估模块
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.train_data = load_data(config["train_data_path"], config, shuffle=False)
        self.schema = self.train_data.dataset.schema  # {意图名: id}
        self.id_to_intent = {v: k for k, v in self.schema.items()}  # {id: 意图名}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _precompute_knwb_vectors(self):
        """预计算知识库中所有问题的向量"""
        self.knwb_vectors = []  # 所有问题的向量
        self.question_to_intent = []  # 问题对应的意图id

        # 遍历知识库
        for intent_id, questions in self.train_data.dataset.knwb.items():
            for q in questions:
                with torch.no_grad():
                    q = q.to(self.device).unsqueeze(0)  # 添加批次维度 [1, max_length]
                    vec = self.model(q)  # 编码后应该是 [1, hidden_size]
                    # 确保向量是二维的
                    if vec.dim() == 1:
                        vec = vec.unsqueeze(0)  # 如果被意外压缩，恢复维度
                    self.knwb_vectors.append(vec.cpu())  # 保存到CPU以避免GPU内存溢出
                    self.question_to_intent.append(intent_id)

        # 转为张量
        self.knwb_vectors = torch.cat(self.knwb_vectors, dim=0)  # [n_knwb, hidden_size]
        # L2归一化
        self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        # 移回设备
        self.knwb_vectors = self.knwb_vectors.to(self.device)

    def eval(self, epoch):
        self.logger.info(f"第 {epoch} 轮模型评估开始")
        self.model.eval()
        self._precompute_knwb_vectors()  # 预计算知识库向量

        correct = 0
        total = 0

        for batch in self.valid_data:
            # 将数据移到正确的设备
            input_ids, true_labels = batch
            input_ids = input_ids.to(self.device)
            true_labels = true_labels.to(self.device)

            batch_size = input_ids.shape[0]
            total += batch_size

            # 获取测试句向量
            with torch.no_grad():
                test_vectors = self.model(input_ids)  # 期望形状: [batch_size, hidden_size]

                # 关键检查：确保测试向量是二维的
                if test_vectors.dim() == 1:
                    # 如果被意外压缩，增加一个维度
                    self.logger.warning(f"测试向量维度异常，期望2D但得到1D，形状: {test_vectors.shape}")
                    test_vectors = test_vectors.unsqueeze(0)

                # 确保测试向量形状正确
                if test_vectors.shape[1] != self.config["hidden_size"]:
                    self.logger.error(f"测试向量维度不匹配: {test_vectors.shape}")
                    continue

                # L2归一化
                test_vectors = torch.nn.functional.normalize(test_vectors, dim=-1)

            # 计算与知识库所有向量的相似度
            similarities = torch.matmul(test_vectors, self.knwb_vectors.T)  # [batch_size, n_knwb]

            # 找最相似的知识库问题
            _, top_idx = torch.max(similarities, dim=1)  # [batch_size]

            # 映射到意图id并比较
            for i in range(batch_size):
                pred_intent_id = self.question_to_intent[top_idx[i].item()]
                true_intent_id = true_labels[i].item()
                if pred_intent_id == true_intent_id:
                    correct += 1

        # 计算准确率
        accuracy = correct / total if total > 0 else 0
        self.logger.info(f"验证集准确率: {accuracy:.4f} (正确: {correct}, 总样本: {total})")
        self.logger.info("------------------------")


if __name__ == "__main__":
    from config import Config
    from model import SiameseNetwork
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    model = SiameseNetwork(Config)
    evaluator = Evaluator(Config, model, logger)
    # 注意：需要先加载模型权重才能进行评估
    # model.load_state_dict(torch.load(os.path.join(Config["model_path"], "final_model.pth")))
    # evaluator.eval(10)  # 评估第10轮模型