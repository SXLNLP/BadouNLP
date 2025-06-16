#-*-coding:utf-8-*-
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import dataGenerator, util
from homeworkNet import HomeworkNet

def predict(model_path, input_val, output_val):
    input_size = len(input_val[0])
    output_size = len(output_val[0])
    val_size = len(input_val)
    correct = 0
    incorrect = 0
    log = []
    model = HomeworkNet(input_size, output_size)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        pre_y = model.forward(torch.FloatTensor(input_val))
        for y, y_gt in zip(pre_y, output_val):
            y_class = np.argmax(y)
            y_gt_class = np.argmax(y_gt)
            log.append([y_gt_class, y_class])
            print("org val:{}, predict val:{}, org class:{}, predict class:{}".format(y_gt, y, y_gt_class, y_class))
            if y_class == y_gt_class:
                correct += 1
            else:
                incorrect += 1
        print("predict done==input:{} arrays, predict correct:{}, "
              "predict incorrect:{}".format(val_size, correct, incorrect))
        plt.scatter(range(val_size), [gt for gt, _ in log], label="org")
        plt.scatter(range(val_size), [pre for _, pre in log], marker="*", label="predict")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    x, y = dataGenerator.make_datas(dataGenerator.generator_muti_max_data, 5, 10)
    predict("model.bin", x, y)