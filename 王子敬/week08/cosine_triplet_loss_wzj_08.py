import torch
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


def main(config):
    # 创建报错模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = SiameseNetwork(config)
    # 标识是否使用GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，迁移模型至GPU")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"epoch: {epoch} begin")
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            # input_id1, input_id2, labels = batch_data  # 输入变化时这里需要修改
            input_id1, input_id2, input_id3 = batch_data  # 输入变化时这里需要修改
            # loss = model(input_id1, input_id2, labels)
            loss = model(input_id1, input_id2, input_id3)
            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info(f"batch loss: {loss}")
            loss.backward()
            optimizer.step()
        logger.info(f"epoch: {epoch} end")
        logger.info(f"epoch average loss: {np.mean(train_loss)}")
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    main(Config)

 # 以下是model中修改的部分
    # def forward(self, sentence1, sentence2=None, sentence3=None, target=None):
    #     # 传入三元组时（sentence1 = anchor， sentence2 = positive， sentence3 = negative)
    #     if sentence2 is not None and sentence3 is not None:
    #         anchor = self.sentence_encoder(sentence1)
    #         positive = self.sentence_encoder(sentence2)
    #         negative = self.sentence_encoder(sentence3)
    #         return self.cosine_triplet_loss(anchor, positive, negative)
    #     # 同时传入两个句子
    #     if sentence2 is not None:
    #         vector1 = self.sentence_encoder(sentence1)  # vec:(batch_size, hidden_size)
    #         vector2 = self.sentence_encoder(sentence2)
    #         # 如果有标签，则计算loss
    #         if target is not None:
    #             return self.loss(vector1, vector2, target.squeeze())
    #         # 如果无标签，计算余弦距离
    #         else:
    #             return self.cosine_distance(vector1, vector2)
    #     # 单独传入一个句子时，认为正在使用向量化能力
    #     else:
    #         return self.sentence_encoder(sentence1)
