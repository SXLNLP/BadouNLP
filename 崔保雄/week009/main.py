# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from predict import Predict

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)

    #打印出模型参数
    show_model_state_dict(model)

    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    model_path = os.path.join(config["model_path"], f"ner_{config["model_type"]}.pth")
    torch.save(model.state_dict(), model_path)
    return model, train_data

""" 打印出模型参数 """
def show_model_state_dict(model):
    print(f"模型的参数列表：")
    for key, value in model.state_dict().items():
        print(f"参数名称：{key}，参数形状：{value.shape}")
    # print("crf_layer.start_transitions：\n", model.state_dict()["crf_layer.start_transitions"])
    # print("crf_layer.end_transitions：\n", model.state_dict()["crf_layer.end_transitions"])
    # print("crf_layer.transitions：\n", model.state_dict()["crf_layer.transitions"])

if __name__ == "__main__":
    #打印出模型参数
    # train_data = load_data(Config["train_data_path"], Config)
    # model = TorchModel(Config)
    # show_model_state_dict(model)
    # model.load_state_dict(torch.load(model_path))

    """ 以下是进行模型训练，或者预测 """
    is_train = True #是训练模型，还是使用训练好的模型预测
    if is_train:
        """ 训练模型 """
        model, train_data = main(Config)
    else:
        """ 使用已经训练好的模型进行预测 """
        predict = Predict()
        text = "全国政协委员、清华大学中国与世界经济研究中心主任李稻葵指出，2018年中国面临三大攻坚战中，风险防控是第一位的。"
        output = predict.eval(text)
        print(output)



