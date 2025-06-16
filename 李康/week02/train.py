#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import dataGenerator, util
from homeworkNet import HomeworkNet

def evaluate(model, eval_x, eval_y):
    model.eval()
    correct = 0
    wrong = 0
    with torch.no_grad():
        y_pre = model(eval_x)
        for y_gt, y_p in zip(eval_y, y_pre):
            if util.is_same_class(y_gt, y_p):
                correct +=1
            else:
                wrong += 1
        return correct / (wrong + correct)

def train():
    epochs = 50
    lr = 1e-3
    input_size= 5
    output_size = 5
    train_data_num = 1000
    evaluate_data_num = 1000
    batch_size = 5
    loss_each_batch = []
    log = []

    train_x, train_y = dataGenerator.make_datas(dataGenerator.generator_muti_max_data, input_size, train_data_num)
    evaluate_x, evaluate_y = dataGenerator.make_datas(dataGenerator.generator_muti_max_data,
                                                      input_size, evaluate_data_num)
    util.print_class_num("eval set include:", evaluate_y)
    model = HomeworkNet(input_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for batch_index in range(train_data_num // batch_size):
            print("============start train epoch{}, batch{}".format(epoch, batch_index))
            x_batch = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y_batch = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x_batch, y_batch)
            loss_each_batch.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("======end train epoch{}, batch{}, loss{}".format(epoch, batch_index, loss.item()))
        # evaluate
        acc = evaluate(model, evaluate_x, evaluate_y)
        log.append([acc, np.mean(loss_each_batch)])
        print("epoch:{} eval acc:{}".format(epoch, acc))
    # draw
    plt.plot(range(len(log)), [acc for acc, _ in log], label="acc")
    plt.plot(range(len(log)), [loss for _, loss in log], label="loss")
    plt.legend()
    plt.show()
    # save model
    torch.save(model.state_dict(), "model.bin")

if __name__ == '__main__':
    train()


