import torch
import torch.nn as nn

from bert_ner.evaluate import Evaluator
from bert_ner.loader import DataGenerator, load_data
from config import Config
from model import NerBertModel, choose_optimizer
import logging
import random
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载数据集
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = NerBertModel(config)
    # 是否使用cpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
    # 选择优化
    optimizer = choose_optimizer(config, model)
    # 加载效果测试
    evaluator = Evaluator(config, model, logger)
    # 训练
    for epoch in range(config["epochs"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data
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
