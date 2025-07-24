

from transformers import BertModel


if __name__ == '__main__':
    bert = BertModel.from_pretrained(r"D:\nlp\prc\nlp-prc\week6\review\bert-base-chinese", return_dict=False)
    param_generator = bert.parameters()

    num = 0
    for layer in param_generator:
        num += len(layer)

    print(num) #227466
