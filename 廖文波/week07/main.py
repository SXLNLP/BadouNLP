
import torch
import random
import os
import numpy as np
import logging
import pandas as pd
from datetime import datetime
from config import Config
from data_loader import load_data
from model import TorchModel, choose_optimizer
from evaluate import Evaluator


#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    #标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        # print(train_data.dataset.data[0])  # 打印第一条数据
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            optimizer.zero_grad()
            input_ids, labels = batch_data
            # print("输入内容", input_ids[0], input_ids.shape)
            # print("标签", labels[0], labels.shape)
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step() ## 梯度下降

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(model, epoch)
    # acc = evaluator.eval(model, 1)
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)
    return acc

class ResultsWriter:
    """用于将实验结果写入Excel文件的工具类"""
    
    def __init__(self, file_path="实验结果.xlsx"):
        """初始化结果写入器"""
        self.file_path = file_path
        self.results = []
        
        # 若文件不存在，创建一个新的DataFrame
        if not os.path.exists(file_path):
            self.df = pd.DataFrame(columns=[
                '时间', '模型类型', '学习率', '隐藏层大小', 
                '批次大小', '池化方式', '准确率'
            ])
            self.save()
    
    def add_result(self, config, accuracy):
        """添加一组实验结果"""
        result = {
            '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '模型类型': config["model_type"],
            '学习率': config["learning_rate"],
            '隐藏层大小': config["hidden_size"],
            '批次大小': config["batch_size"],
            '池化方式': config["pooling_style"],
            '准确率': accuracy
        }
        self.results.append(result)
    
    def save(self):
        """将所有结果保存到Excel文件"""
        # 读取已有的数据
        if os.path.exists(self.file_path):
            self.df = pd.read_excel(self.file_path)
        
        # 添加新数据
        if self.results:
            new_df = pd.DataFrame(self.results)
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            
            # 保存到Excel
            with pd.ExcelWriter(self.file_path, engine='openpyxl', mode='w') as writer:
                self.df.to_excel(writer, index=False)
            
            # 清空临时结果列表
            self.results = []
            print(f"已将 {len(new_df)} 条结果保存到 {self.file_path}")

if __name__ == "__main__":
    # main(Config)
    # 初始化结果写入器
    results_writer = ResultsWriter()

    print("开始搜索超参数")
    for model in ["rnn","lstm","bert"]:
        print("当前模型：", model)
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            print("当前学习率：", lr)
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                print("当前隐藏层大小：", hidden_size)
                Config["hidden_size"] = hidden_size
                for batch_size in [128]:
                    Config["batch_size"] = batch_size
                    print("当前批量大小：", batch_size) 
                    for pooling_style in ["max","avg"]:
                        Config["pooling_style"] = pooling_style
                        # 执行实验
                        accuracy = main(Config)
                        # 记录结果
                        print("当前池化样式：", pooling_style, "当前准确率：", accuracy)
                        results_writer.add_result(Config, accuracy)
                        # 打印当前结果
                        print("最后一轮准确率：", accuracy, "当前配置：", Config)

    # 保存所有结果到Excel
    results_writer.save()