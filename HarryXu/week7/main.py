import torch
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

'''
模型训练主程序
'''

seed = Config['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config['model_path']):
        os.mkdir(config['model_path'])
    # 加载训练数据
    train_data = load_data(config['train_data_path'], config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info('gpu可以使用，迁移模型至gpu')
        model = model.cuda()
    # 优化器
    optimizer = choose_optimizer(config, model)
    # 加载测评类
    evaluator = Evaluator(config, model, logger)
    # 训练
    model.train()
    for epoch in range(config['epoch']):
        logger.info('epoch %d begin' % (epoch + 1))
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            optimizer.zero_grad()
            input_ids, labels = batch_data
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info('batch loss %f' % loss)
        logger.info('epoch average loss: %f' % np.mean(train_loss))
        acc = evaluator.eval(epoch + 1)
        time = evaluator.get_time()
    return acc, time


if __name__ == '__main__':
    with open(Config['model_path'] + './result.txt', 'w') as file:
        for model in ['gated_cnn', 'rnn', 'lstm', 'gru']:
            Config['model_type'] = model
            for lr in [1e-3, 1e-4]:
                Config['learning_rate'] = lr
                for hidden_size in [128]:
                    Config['hidden_size'] = hidden_size
                    for batch_size in [64, 128]:
                        Config['batch_size'] = batch_size
                        for pooling_style in ['avg', 'max']:
                            Config['pooling_style'] = pooling_style
                            rate, time = main(Config)
                            file.write(str(rate) + '--' + str(time) + '--' + str(Config) + '\n')
                            print('最后一轮准确率：', rate, '当前配置：', Config)