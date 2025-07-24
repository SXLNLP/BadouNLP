import torch
import os
import random
import numpy as np
import logging
import time
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from tabulate import tabulate
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(filename="my_logs.log", filemode="w", level=logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
    # 加载模型
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
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            if model.use_bert:
                input_ids, attention_mask, labels = batch_data
                loss = model(input_ids, labels, attention_mask=attention_mask)
            else:
                input_ids, labels = batch_data
                loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        if epoch < config["epoch"]:
            acc = evaluator.eval(epoch)
        elif epoch == config["epoch"]:
            start_time = time.time()
            acc = evaluator.eval(epoch)
            end_time = time.time()
            spends = end_time - start_time
        
    model_path = os.path.join(config["model_path"], "%s.pth" % config["model_type"])
    torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc, spends

# 对比不同模型、不同参数的效果
def models_contrast(model, lr, batch_size, epoch, pooling_style, acc, spends, hidden_size=768):
    if model == "bert" or model == "bert_cnn" or model == "bert_lstm" or model == "bert_mid_layer":
        return {"model":model, "lr":lr, "batch_size":batch_size, "epoch":epoch, "pooling_style":pooling_style, "hidden_size":768, "acc":f"{acc:.3f}", "spends":f"{spends:.4f}"}
    return {"model":model, "lr":lr, "batch_size":batch_size, "epoch":epoch, "pooling_style":pooling_style, "hidden_size":hidden_size, "acc":f"{acc:.3f}", "spends":f"{spends:.4f}"}


if __name__ == "__main__":
    # main(Config)

    # 对比所有模型
    # 超参数的网格搜索
    contrast = []
    for model in ["gated_cnn", 'bert', 'lstm', 'bert_lstm']:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for batch_size in [64, 128]:
                Config["batch_size"] = batch_size
                for epoch in [5, 10]:
                    Config["epoch"] = epoch
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        if model not in ("bert", "bert_cnn", "bert_lstm", "bert_mid_layer"):
                            for hidden_size in [128, 256]:
                                Config["hidden_size"] = hidden_size
                                acc, spends = main(Config)
                                print("最后一轮准确率：", acc, "当前配置：", Config)
                                contrast.append(models_contrast(model, lr, batch_size, epoch, pooling_style, acc, spends, hidden_size))
                        else:
                            acc, spends = main(Config)
                            print("最后一轮准确率：", acc, "当前配置：", Config)
                            contrast.append(models_contrast(model, lr, batch_size, epoch, pooling_style, acc, spends))
    print(tabulate(contrast, headers="keys", tablefmt="grid"))


