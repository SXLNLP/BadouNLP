# -*- coding: utf-8 -*-
import torch
from loader import load_data
import time

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _sync_cuda(self):
        """同步CUDA操作，确保时间测量准确"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果

        # 初始化计时器
        start_time = time.perf_counter()
        total_samples = 0
        batch_start = time.perf_counter()

        # 正式计时
        self.logger.info("开始计时...")

        batch_time_log=[]
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
            total_samples += labels.size(0)

            # 记录批次耗时
            batch_end = time.perf_counter()
            batch_time = batch_end - batch_start
            batch_time_log.append(batch_time)
            #self.logger.info(f"Batch {index + 1:04d} | 耗时: {batch_time:.3f}s | "
                             #f"累计耗时: {batch_end - start_time:.3f}s | "
                             #f"吞吐量: {total_samples / (batch_end - start_time):.2f} samples/s")
            batch_start = batch_end  # 重置批次起始时间

        # 最终统计

        total_time = time.perf_counter() - start_time
        avg_throughput = total_samples / total_time
        avg_batch_time = total_time / len(batch_time_log)
        self.logger.info("===== 验证完成 =====")
        self.logger.info(f"总样本数: {total_samples}")
        self.logger.info(f"总耗时: {total_time:.3f}s")
        self.logger.info(f"batch个数: {len(batch_time_log)}")
        self.logger.info(f"平均batch耗时: {avg_batch_time:.3f}s")
        self.logger.info(f"平均吞吐量: {avg_throughput:.2f} samples/s")

        acc = self.show_stats()
        return acc, total_time

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
