# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

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
    model = SiameseNetwork(config)
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

            input_id1, input_id2, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况

            """ 【作业修改代码】判断是否使用TripletLoss损失函数 """
            if config["loss_type"] == "TripletLoss":
                #使用TripletLoss当损失函数时，labels并不是标签，而是跟input_id1, input_id2一样，都代表一个句子。三个分别对应TripletLoss(a,p,n)中的a,p,n
                loss = model(input_id1, input_id2, labels, None)
            else:
                loss = model(input_id1, input_id2, None, labels)

            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        logger.info(f"损失函数类型：{config["loss_type"]}")
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":
    main(Config)

    """ 加载模型，读出模型参数。前面模型训练完保存的是state_dict()，即模型权重参数 """
    # path = Config["model_path"] + "/epoch_10.pth"
    # model = torch.load(path)
    #
    # for param in model.items():
    #     param_name, param_value = param
    #     print("模型参数：",param_name, "，形状：", param_value.shape)

    """
        sentence_encoder.embedding.weight torch.Size([4623, 128])
        sentence_encoder.layer.weight torch.Size([128, 128])
        sentence_encoder.layer.bias torch.Size([128])
    """
