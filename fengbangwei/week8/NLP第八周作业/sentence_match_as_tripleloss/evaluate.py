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
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # 由于效果测试需要训练集当做知识库，再次加载训练集。
        # 事实上可以通过传参把前面加载的训练集传进来更合理，但是为了主流程代码改动量小，在这里重新加载一遍
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    # 将知识库中的问题向量化，为匹配做准备
    # 每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
    def knwb_to_vector(self):
        # {
        #     0: 0,  # q1 属于标准问 0（套餐变更）
        #     1: 0,  # q2 属于标准问 0
        #     2: 1,  # q3 属于标准问 1（话费查询）
        #     3: 1,  # q4 属于标准问 1
        #     4: 1   # q5 属于标准问 1
        # }
        self.question_index_to_standard_question_index = {}
        # self.question_ids = [q1, q2, q3, q4, q5]
        # q1 = torch.LongTensor([1, 2, 3, 4])  # 第一个问题，长度为4
        # q2 = torch.LongTensor([5, 6, 7, 0])  # 第二个问题，长度为4，最后一个为 padding
        # q3 = torch.LongTensor([8, 9, 0, 0])  # 第三个问题，长度为4，最后两个为 padding
        self.question_ids = []
        # self.knwb[self.schema[label]].append(input_id) {"0": [标准问1, 标准问2, ...], "1": [标准问1, 标准问2, ...], ...}
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                # 记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            # 将self.question_ids中的所有question_id按维度0进行堆叠，生成一个question_matrixs张量
            #question_matrixs 输出：
            #tensor([[1, 2, 3, 4],
            #        [5, 6, 7, 0],
            #        [8, 9, 0, 0]])
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
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
            input_id, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                test_question_vectors = self.model(input_id)  # 不输入labels，使用模型当前参数进行预测
            self.write_stats(test_question_vectors, labels)
        self.show_stats()
        return

    def write_stats(self, test_question_vectors, labels):
        assert len(labels) == len(test_question_vectors)
        for test_question_vector, label in zip(test_question_vectors, labels):
            # 通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            # test_question_vector shape [vec_size]   knwb_vectors shape = [n, vec_size]
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
            hit_index = self.question_index_to_standard_question_index[hit_index]  # 转化成标准问编号
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
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return
