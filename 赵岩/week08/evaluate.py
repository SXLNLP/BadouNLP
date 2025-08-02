# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        # 加载验证集
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # 重新加载训练集作为知识库
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    # 将知识库中的问题向量化，为匹配做准备
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                # 记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)  # 注意这里model输出可能需要调整以适应新结构
            # 将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空前一轮的测试结果
        self.model.eval()
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data  # 假设输入仍为 (query, label) 形式
            with torch.no_grad():
                test_question_vectors = self.model(input_id)  # 这里假设只输入了 query 部分
            self.write_stats(test_question_vectors, labels)
        self.show_stats()
        return

    def write_stats(self, test_question_vectors, labels):
        assert len(labels) == len(test_question_vectors)
        for test_question_vector, label in zip(test_question_vectors, labels):
            # 通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
            hit_index = self.question_index_to_standard_question_index[hit_index]  # 转换成标准问编号
            if int(hit_index) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        accuracy = correct / (correct + wrong) if (correct + wrong) > 0 else 0
        self.logger.info("预测准确率：%f" % accuracy)
        self.logger.info("--------------------")
        return
