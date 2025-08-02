# -*- coding: utf-8 -*-
import torch  # 导入PyTorch深度学习框架
import os  # 导入操作系统接口模块，用于文件和目录操作
import random  # 导入随机数模块，用于数据打乱等操作
import numpy as np  # 导入numpy库，用于数值计算
import logging  # 导入日志模块，用于记录训练过程信息
from config import Config  # 导入配置模块
from model import TorchModel, choose_optimizer  # 导入模型和优化器选择函数
from evaluate import Evaluator  # 导入评估器类
from loader import load_data  # 导入数据加载函数

# 配置日志记录格式
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)  # 获取日志记录器

"""
模型训练主程序
"""

def main(config):  # 主函数
    # 如果模型保存目录不存在，则创建该目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 创建模型实例
    model = TorchModel(config)
    # 检查GPU是否可用
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:  # 如果GPU可用
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()  # 将模型迁移到GPU
    # 选择优化器
    optimizer = choose_optimizer(config, model)
    # 创建评估器实例
    evaluator = Evaluator(config, model, logger)
    # 训练循环
    for epoch in range(config["epoch"]):  # 遍历每个训练轮次
        epoch += 1  # 轮次计数从1开始
        model.train()  # 设置模型为训练模式
        logger.info("epoch %d begin" % epoch)  # 记录轮次开始
        train_loss = []  # 存储本轮次的损失值
        # 遍历每个批次的数据
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()  # 清空梯度
            if cuda_flag:  # 如果使用GPU
                batch_data = [d.cuda() for d in batch_data]  # 将数据迁移到GPU
            # 解包批次数据
            input_id, labels = batch_data
            # 前向传播，计算损失
            loss = model(input_id, labels)
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新参数
            train_loss.append(loss.item())  # 记录损失值
            # 定期输出损失信息
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        # 记录本轮次平均损失
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)  # 评估模型效果
    # 构建模型保存路径
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # 保存模型状态（当前被注释掉了）
    # torch.save(model.state_dict(), model_path)
    return model, train_data  # 返回模型和训练数据

if __name__ == "__main__":  # 程序入口
    model, train_data = main(Config)  # 运行主函数
