# -*- coding: utf-8 -*-
import time

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
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)

    def test_speed(self, num_samples=100):
        """测试模型预测速度"""
        self.model.eval()
        total_time = 0
        samples_collected = 0
        test_data = []

        # 从DataLoader迭代器中收集指定数量的样本
        for batch_data in self.valid_data:
            if samples_collected >= num_samples:
                break

            test_data.append(batch_data)
            samples_collected += len(batch_data[0])  # 假设batch_data[0]是输入数据

        # 如果收集的样本超过需求，裁剪最后一个batch
        if samples_collected > num_samples:
            input_ids, labels = test_data[-1]
            # 假设batch_data[0]是输入，batch_data[1]是标签
            test_data[-1] = (
                input_ids[:num_samples - (samples_collected - len(input_ids))],
                labels[:num_samples - (samples_collected - len(input_ids))]
            )

        with torch.no_grad():
            for batch_data in test_data:
                if torch.cuda.is_available():
                    batch_data = [d.cuda() for d in batch_data]

                input_ids, _ = batch_data
                start = time.time()
                _ = self.model(input_ids)
                end = time.time()

                # 累计批处理时间
                total_time += (end - start)

        # 计算平均每条耗时(毫秒)
        avg_time = (total_time / num_samples) * 1000
        return avg_time
