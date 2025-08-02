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
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return model, train_data

if __name__ == "__main__":
    model, train_data = main(Config)

# 2025-07-15 07:17:56,225 - __main__ - INFO - --------------------
# 2025-07-15 07:17:56,225 - __main__ - INFO - epoch 20 begin
# 2025-07-15 07:17:56,412 - __main__ - INFO - batch loss 0.006332
# 2025-07-15 07:18:03,736 - __main__ - INFO - batch loss 0.004352
# 2025-07-15 07:18:10,886 - __main__ - INFO - batch loss 0.003551
# 2025-07-15 07:18:10,886 - __main__ - INFO - epoch average loss: 0.006158
# 2025-07-15 07:18:10,886 - __main__ - INFO - 开始测试第20轮模型效果：
# 2025-07-15 07:18:12,262 - __main__ - INFO - PERSON类实体，准确率：0.633333, 召回率: 0.489691, F1: 0.552321
# 2025-07-15 07:18:12,262 - __main__ - INFO - LOCATION类实体，准确率：0.635593, 召回率: 0.627615, F1: 0.631574
# 2025-07-15 07:18:12,262 - __main__ - INFO - TIME类实体，准确率：0.888158, 召回率: 0.758427, F1: 0.818177
# 2025-07-15 07:18:12,262 - __main__ - INFO - ORGANIZATION类实体，准确率：0.573333, 召回率: 0.452632, F1: 0.505877
# 2025-07-15 07:18:12,262 - __main__ - INFO - Macro-F1: 0.626987
# 2025-07-15 07:18:12,264 - __main__ - INFO - Micro-F1 0.641390
# 2025-07-15 07:18:12,264 - __main__ - INFO - --------------------

# 使用Bert
    # "max_length": 100,
    # "num_layers": 2,
    # "epoch": 20,
    # "batch_size": 16,
    # "optimizer": "adam",
    # "learning_rate": 1e-5,
# 2025-07-18 21:45:50,907 - __main__ - INFO - --------------------
# 2025-07-18 21:45:50,909 - __main__ - INFO - epoch 20 begin
# 2025-07-18 21:45:52,319 - __main__ - INFO - batch loss 3.932858
# 2025-07-18 21:46:49,765 - __main__ - INFO - batch loss 5.023751
# 2025-07-18 21:47:44,854 - __main__ - INFO - batch loss 0.416180
# 2025-07-18 21:47:44,855 - __main__ - INFO - epoch average loss: 5.246906
# 2025-07-18 21:47:44,863 - __main__ - INFO - 开始测试第20轮模型效果：
# 2025-07-18 21:47:54,817 - __main__ - INFO - PERSON类实体，准确率：0.466667, 召回率: 0.252577, F1: 0.327755
# 2025-07-18 21:47:54,817 - __main__ - INFO - LOCATION类实体，准确率：0.662338, 召回率: 0.426778, F1: 0.519079
# 2025-07-18 21:47:54,817 - __main__ - INFO - TIME类实体，准确率：0.823077, 召回率: 0.601124, F1: 0.694800
# 2025-07-18 21:47:54,817 - __main__ - INFO - ORGANIZATION类实体，准确率：0.534483, 召回率: 0.326316, F1: 0.405224
# 2025-07-18 21:47:54,818 - __main__ - INFO - Macro-F1: 0.486715
# 2025-07-18 21:47:54,818 - __main__ - INFO - Micro-F1 0.501296
# 2025-07-18 21:47:54,818 - __main__ - INFO - --------------------