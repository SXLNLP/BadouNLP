import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
import time
import csv

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    # 1、创建保存模型训练结果的目录、保存模型结果的目录
    if not os.path.isdir(config["result_compare_path"]):
        os.mkdir(config["result_compare_path"])
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 2、加载训练数据：根据字（而非分词）来构建词向量，是torch数据批处理加载类型
    train_data, valid_data = load_data(config["data_path"], config)
    config['train_data_size'] = len(train_data.dataset)
    config['valid_data_size'] = len(valid_data.dataset)
    # 3、加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 4、加载优化器: adam、sgd
    optimizer = choose_optimizer(config, model)
    # 5、加载验证数据集测试效果：不同的模型/学习率 来进行比较
    evaluator = Evaluator(config, model, logger, valid_data)
    train_time = 0
    valid_time = 0
    # 6、开始epoch多次训练
    logger.info("******************************************************************")
    logger.info("%s model start training, optimizer: %s, learning rate: %f" %
                (config["model_type"], config["optimizer"], config["learning_rate"]))
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d start......" % epoch)
        start_time = time.time()
        train_loss = []
        batch_i = 0
        for index, batch_data in enumerate(train_data):
            batch_i += 1
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch %d loss: %f" % (batch_i, loss))

        train_time += time.time() - start_time
        start_time = time.time()
        logger.info("epoch %d average loss: %f" % (epoch, np.mean(train_loss)))
        acc = evaluator.eval(epoch)
        valid_time += time.time() - start_time

        model_path = os.path.join(config["model_path"], "%s_%s_%f_model_%d.pth" %
                                  (config['model_type'], config['optimizer'], config['learning_rate'], config["epoch"]))
        torch.save(model.state_dict(), model_path)  # 保存模型权重
    return acc, train_time, valid_time


if __name__ == "__main__":
    # 超参数的网格搜索
    headers = ["Model", "Learning_Rate", "Hidden_Size", "Epoch", "Batch_Size", "Train_Data_Size", "Valid_Data_Size",
               "Vacob_Size", "Acc", "Train_Time(秒/1轮)", "Valid_Time(秒/1轮)", "Predict_Time(秒/100条)"]
    learnging_rate = [0.001, 0.01, 0.1]
    models = ["rnn", "lstm", 'cnn', 'rcnn', 'gated_cnn', 'gru', 'stack_gated_cnn', 'fast_text']

    csv_data = []
    # 根据不同的模型和参数进行训练比较
    for model in models:
        Config["model_type"] = model
        for lr in learnging_rate:
            Config["learning_rate"] = lr
            acc, train_time, valid_time = main(Config)
            row = []
            row.append(model)
            row.append(Config['learning_rate'])
            row.append(Config['hidden_size'])
            row.append(Config['epoch'])
            row.append(Config['batch_size'])
            row.append(Config['train_data_size'])
            row.append(Config['valid_data_size'])
            row.append(Config['vocab_size'] + 1)
            row.append(acc)
            row.append(train_time / Config['epoch'])
            row.append(valid_time / Config['epoch'])
            row.append(valid_time / Config['epoch'] / Config['valid_data_size'] * 100)
            csv_data.append(row)
    # 写入CSV文件
    logger.info("start write csv file of {model}'s info and evaluated result......")
    try:
        with open(f"./output/output.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # 写入表头（可选）
            writer.writerow(headers)
            # 写入多行数据
            writer.writerows(csv_data)
            logger.info(f"write {model}'s csv file completed..")
    except (FileNotFoundError, PermissionError) as e:
        logger.info(f"write {model}'s csv file filed.")