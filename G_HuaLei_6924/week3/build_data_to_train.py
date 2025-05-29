import json
import random
import torch
from config import Config


def build_data_to_train(num_classes=Config['num_classes']):
    '''随机生成英文小写26个字母组成的字符串，其中有一半概率包含字母a,字符串长度为5'''
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
    sample = []
    for i in range(num_classes):
        if random.random() < 0.2:
            sample.append('a')
        else:
            sample.append(random.choice('bcdefghijklmnopqrstuvwxyz'))
    sample = [vocab.get(word, vocab['<unk>']) for word in sample]
    if 1 in sample:
        target = sample.index(1)
    else:
        target = num_classes
    return sample, target

def build_dataset(batch_size=Config['batch_size']):
    sample_dataset = []
    target_dataset = []
    for i in range(batch_size):
        sample, target = build_data_to_train(Config['num_classes'])
        sample_dataset.append(sample)
        target_dataset.append(target)

    return torch.LongTensor(sample_dataset), torch.LongTensor(target_dataset)


#测试代码
#用来测试每轮模型的准确率
def evaluate(model, device):
    model.eval()
    x, y = build_dataset(Config['evaluate_batch_size'])
    x, y = x.to(device), y.to(device)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)

        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            max_index = torch.argmax(y_p).int()
            max_value = torch.max(y_p).float()
            if max_value > 0.5 and max_index == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def predict(model, model_path, vocab_path ):
    input_vec = [['a', 'b', 'g', 'g', 'f', 'z'],
                 ['0000', 'b', 'g', 'g', 'f', 'z'],
                 ['dfsdafd', 'b', 'g', 'g', 'f', 'z'],
                 ['l', 'b', '', 'a', 'f', 'z'],
                 ['a', 'b', '正则', 'g', 'f', 'z'],
                 ['y', 'b', 'g', 'g', 'f', 'z'],
                 ['z', 'b', 'g', 'g', '~~', 'a'],
                 ['==', 'b', 'g', 'g', 'f', 'z'],
                 ['n', 'b', 'a', 'g', 'f', 'z'],]
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    input_vec = [[vocab.get(char, vocab['<unk>']) for char in input_string] for input_string in input_vec]
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()
    with torch.no_grad():
        label_pred = model(torch.LongTensor(input_vec))
    for sample, target in zip(input_vec, label_pred):
        print(sample, '===', torch.argmax(target)+1)

if __name__ == '__main__':
    pass
    # build_data_to_train()