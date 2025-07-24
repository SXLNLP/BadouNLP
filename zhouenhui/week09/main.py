import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from transformers import BertForTokenClassification
from torch.utils.data import DataLoader

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

    dataset = load_data(config["train_data_path"], config)
    train_data = DataLoader(dataset, batch_size=config["batch_size"])
    #加载模型
    model = BertForTokenClassification.from_pretrained(config["bert_path"],num_labels=config["class_num"],output_attentions=False,output_hidden_states=False)
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
    for epoch in range(config["epoch"]): #外层循环控制训练轮次
        epoch += 1
        model.train() #每轮开始时将模型设置为训练模式
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad() #清空梯度，防止梯度累积
            if cuda_flag:

                batch_data = {k: v.cuda() for k, v in batch_data.items()}

            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            labels = batch_data["labels"]
            output = model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
            loss = output[0] #前向传播计算损失
            loss.backward() #反向传播d
            optimizer.step() #参数更新
            train_loss.append(loss.item())
            if index % int(len(dataset) / (2*config["batch_size"])) == 0: #每完成半个epoch记录一次batch损失
                logger.info("batch loss %f" % loss) #使用logger记录epoch开始、batch损失和epoch平均损失
        logger.info("epoch average loss: %f" % np.mean(train_loss)) #每轮结束后记录平均损失
        evaluator.eval(epoch) #每轮训练结束后进行模型评估
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch) #将模型按训练轮次(epoch)编号保存到指定目录
    torch.save(model.state_dict(), model_path)
    return model, dataset

if __name__ == "__main__":
    model, dataset = main(Config)
    train_loader = DataLoader(dataset,batch_size=Config["batch_size"])
