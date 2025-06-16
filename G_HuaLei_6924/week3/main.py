import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import Config
from model import TorchModel, choose_optimizer
from build_data_to_train import build_dataset, evaluate, predict


def main(config):
    # 创建保存模型的目录
    # if not os.path.isdir(config["model_path"]):
    #     os.mkdir(config["model_path"])
    # 定义设备（自动选择 GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义模型
    model = TorchModel(config).to(device)
    # 优化器
    optimizer = choose_optimizer(model, config)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        watch_loss = []
        for batch in range(config['sample_amount'] // config["batch_size"]):
            # 获取训练数据
            x, y = build_dataset(config["batch_size"])
            # 将数据转移到设备
            x, y = x.to(device), y.to(device)
            # 模型前向传播
            loss = model(x, y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        print("epoch: %d, loss: %f" % (epoch, np.mean(watch_loss)))
        # 评测模型正确率
        evaluate(model, device)
        # 保存模型
        torch.save(model.state_dict(), config["model_path"])




if __name__ == "__main__":
    # main(Config)

    # 模型预测
    model = TorchModel(Config)
    predict(model, Config['model_path'], Config['vocab_path'])