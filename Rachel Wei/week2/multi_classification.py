import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
Project Prompt:

Use Pytorch framework, build a program that solves a 5-class classification problem. 
The input would be a 5D vector. If the first element is the largest, categorize the output to class 1. If the second is the largest, categorize to class 2. etc. 

"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5) ## linear model could reflect any dimension input to any dimension output
        self.loss = nn.functional.cross_entropy ## use cross entropy as the loss function

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            # print('x ', x)
            # print('y ', y)
            # print('y_pred is ', y_pred)
            return self.loss(y_pred, y)
        else:
            # print('y_pred is ', y_pred)
            return y_pred
        

def find_max(vector):
    max = float('-inf')
    max_marker = -1
    for i in range(5):
        if vector[i] > max:
            max = vector[i]
            max_marker = i
    output = np.zeros(5)
    output[max_marker] = 1
    return vector, output


def build_sample():
    x = np.random.random(5)
    return find_max(x)


def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y) # when using crossentropy with multi-dimension vector, we don't need to mark Y output as a scalar using [Y]
    return torch.FloatTensor(X), torch.FloatTensor(Y) # when using crossentropy, we want Y to be LongTensor

def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    correct, wrong = 0, 0
    with torch.no_grad(): # no gradient calculation to speed up inference. ensure 'forward' function would not be called
        y_pred = model(x)   #output of this step is the tensor of 100 * 1 * 5
        for y_p, y_t in zip(y_pred,y):
            y_p_class = torch.argmax(y_p).item()
            y_t_class = torch.argmax(y_t).item()
            if y_p_class == y_t_class:
                correct += 1
            else:
                wrong += 1
    print("correct count: %d, correct rate: %f" % (correct, correct/(correct + wrong)))
    return correct / (correct + wrong)
    

def main():
    epoch_num = 100 # total number of epoch
    train_sample = 5000 # sample size of each epoch
    batch_size = 10 # sample size of each batch within each epoch
    input_size = 5 # dimension of input
    learning_rate = 0.001 #learning rate
    model = TorchModel(input_size) # build model
    optim = torch.optim.Adam(model.parameters(), lr = learning_rate) # optimizer
    log = []
    train_x, train_y = build_dataset(train_sample) # build up training data

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for ind in range(train_sample//batch_size):
            ind_left = ind * batch_size
            ind_right = (ind+1) * batch_size
            x = train_x[ind_left : ind_right]
            y = train_y[ind_left : ind_right]
            loss = model(x, y) # input the x and y to the model. model will autimatically calculate loss
            loss.backward() # calculate gradient
            optim.step() # update gradient into the model
            optim.zero_grad() # clear gradient to 0
            watch_loss.append(loss.item())
        print("========\n Epoch Number: %d, Average Loss: %f =========" %(epoch+1, np.mean(watch_loss)))
        acc = evaluate(model)
        print(' Epoch Accuracy', acc)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model2.bin")
    plt.plot(range(len(log)), [l[0] for l in log], label = 'acc')
    plt.plot(range(len(log)), [l[1] for l in log], label = 'loss')
    plt.legend()
    plt.show()
    return

def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print('im here')
    print(model.state_dict())
    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec)) # output is a tensor of the results of predicted Y
        print('predicted result step 1', result)
    for vec, res in zip(input_vec, result):
        class_output = torch.argmax(res).item()
        print("input vector is: %s, model classification is :%d" % (vec, class_output))  

if __name__ == "__main__":
    main()
    test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model2.bin", test_vec)


