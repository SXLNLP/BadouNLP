import os.path

import numpy as np
import torch
import torch.nn as nn


class ClassificationModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(ClassificationModel, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_pred = self.linear(x)

        if y is None:
            return torch.argmax(y_pred, dim=1, keepdim=True)
        else:
            return self.loss(y_pred, y)


class DataSet:
    def __init__(self, dim, sample_num):
        super(DataSet).__init__()
        self.dim = dim
        self.sample_num = sample_num

    def __get_sample(self):
        x = np.random.random(self.dim)
        return x, np.argmax(x)  # sample, 分类

    def get_sample(self):
        X = []
        cls = []
        for i in range(self.sample_num):
            x, cl = self.__get_sample()
            X.append(x)
            cls.append(cl)

        return X, self.list_to_one_hot(cls)

    def list_to_one_hot(self, cls: list):
        one_hot = np.zeros((len(cls), self.dim), dtype=np.float32)
        for i, v in enumerate(cls):
            one_hot[i][v] = 1.0

        return one_hot


class Trainer:
    def __init__(self,
                 epoch_num,
                 sample_num,
                 sample_dim,
                 batch_size,
                 learning_rate,
                 model: nn.Module,
                 optimizer
                 ):
        self.epoch_num = epoch_num
        self.sample_num = sample_num
        self.sample_dim = sample_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = model
        self.optimizer_name = optimizer
        self.__setup_optimizer()

    def __setup_optimizer(self):
        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):

        dataset = DataSet(self.sample_dim, self.sample_num)
        self.model.train()

        epoch = 0
        log = []
        while epoch < self.epoch_num:
            max_batch_num = self.sample_num // self.batch_size

            x, y = dataset.get_sample()

            batch_index = 0

            watch_loss = []

            while batch_index < max_batch_num:
                x_train = x[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
                y_train = y[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

                loss = self.model(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_index += 1

                watch_loss.append(loss.detach().numpy())
            epoch += 1

            print('-' * 10, f"epoch: {epoch},  loss: {np.mean(watch_loss)}", '-' * 10)

            acc = self.evaluate()
            print("-" * 10, f"acc_rate: {acc}", "-" * 10)
            log.append([acc, float(np.mean(watch_loss))])

    def evaluate(self):
        self.model.eval()
        test_sample_num = 1000
        test_sample = DataSet(self.sample_dim, test_sample_num)

        x, y = test_sample.get_sample()

        with torch.no_grad():
            x_test = torch.FloatTensor(x)

            y_preds = self.model(x_test)

            correct, wrong = 0, 0

            for y_pred, y_t in zip(y_preds.numpy(), np.argmax(y, keepdims=True)):
                if y_pred == y_t:
                    correct += 1
                else:
                    wrong += 1

            rate = correct / (correct + wrong)
            return rate

    def predict(self, test):
        self.model.eval()

        with torch.no_grad():
            return self.model(torch.FloatTensor(test))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


class Predictor:
    __DEFAULT_PATH = "model.bin"
    __DEFAULT_MODEL_PARAM = {
        "epoch_num": 100,
        "sample_num": 500,
        "sample_dim": 5,
        "batch_size": 6,
        "learning_rate": 0.001,
        "optimizer": 'adam',
        "save_path": __DEFAULT_PATH,
        "save": False
    }

    def __init__(self,
                 model: nn.Module = None,
                 path=None):
        super(Predictor).__init__()
        self.dict_path = path
        self.model = model

    def predict(self, p_lst, **kwargs):
        if p_lst is None or len(p_lst) == 0 or len(p_lst[0]) == 0:
            raise Exception("empty predict list")
        default_len = len(p_lst[0])

        if self.model is not None:
            return self.do_predict(self.model, p_lst)
        else:
            kwargs["sample_dim"] = default_len
            self.build_model(**kwargs)
            return self.do_predict(self.model, p_lst)

    def do_predict(self, model, p_lst):
        if model is not None:
            with torch.no_grad():
                return self.model(torch.FloatTensor(p_lst))
        return None

    def build_model(self, **kwargs):
        param = dict()
        param.update(Predictor.__DEFAULT_MODEL_PARAM)
        param.update(kwargs)
        dim = param["sample_dim"]
        self.model = ClassificationModel(dim, dim)

        full_path = self.dict_path

        if self.dict_path is None or len(str.strip(full_path)) == 0:
            self.dict_path = Predictor.__DEFAULT_PATH
            full_path = os.path.join(os.path.dirname(__file__), self.dict_path)

        if os.path.exists(full_path):
            self.model.load_state_dict(torch.load(full_path, weights_only=True))
            return

        trainer = Trainer(param["epoch_num"],
                          param["sample_num"],
                          param["sample_dim"],
                          param["batch_size"],
                          param["learning_rate"],
                          self.model,
                          param["optimizer"])
        trainer.train()
        if param["save"]:
            trainer.save_model(param["save_path"])


def main():
    p = Predictor()
    print(p.predict([[0.1, 0.5, 10, 3, 3.5]], save=True))
    print(p.predict([[4, 0.5, 1, 3, 3.5]], save=True))
    print(p.predict([[4, 0.5, 1, -6, 3.5]], save=True))
    print(p.predict([[3, 8, 1, -6, 3.5]], save=True))
    print(p.predict([[3, 8, 8, -6, 3.5]], save=True))
    print(p.predict([[3, 0, 2.9, -6, 3.5]], save=True))


if __name__ == '__main__':
    main()
