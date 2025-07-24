# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/5/22
# @Author      : liuboyuan
# @Description :
from typing import List, Tuple, Callable, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules.loss import _Loss


class ClassifierTrainer:
    def __init__(
        self,
        input_size: int = 5,
        classify_cond: Callable[[np.ndarray], int] = lambda x: 0,
        learning_rate: float = 0.001
    ):
        """
        初始化分类器训练器

        :param input_size: 输入特征维度
        :param classify_cond: 分类条件函数，输入是5维向量，输出是类别编号
        :param learning_rate: 学习率
        """
        self.input_size = input_size
        self.classify_cond = classify_cond
        self.learning_rate = learning_rate
        self.model: nn.Linear = self._build_model()
        self.optimizer: optim.Adam = self._build_optimizer()

    def _build_model(self) -> nn.Linear:
        """构建线性模型"""
        return nn.Linear(self.input_size, self._get_num_classes())

    def _build_optimizer(self) -> optim.Adam:
        """构建优化器"""
        return optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _get_num_classes(self) -> int:
        """通过模拟调用 classify_cond 推断类别数量"""
        dummy_x = np.random.rand(self.input_size)
        dummy_y = self.classify_cond(dummy_x)
        num_classes = 0
        for _ in range(100):
            y = self.classify_cond(np.random.rand(self.input_size))
            num_classes = max(num_classes, y + 1)
        return num_classes

    def build_sample(self) -> Tuple[np.ndarray, int]:
        """构造单个样本：随机五维向量 + 自定义分类逻辑"""
        x = np.random.rand(self.input_size)
        y = self.classify_cond(x)
        return x, y

    def build_dataset(self, total_sample_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构造数据集

        :param total_sample_num: 样本总数
        :return: 特征张量和标签张量
        """
        X: List[np.ndarray] = []
        Y: List[int] = []
        for _ in range(total_sample_num):
            x, y = self.build_sample()
            X.append(x)
            Y.append(y)
        return torch.FloatTensor(X), torch.LongTensor(Y)

    def evaluate(self, model: Optional[nn.Module] = None, test_sample_num: int = 100) -> float:
        """
        模型评估函数

        :param model: 要评估的模型（默认使用 self.model）
        :param test_sample_num: 测试样本数
        :return: 准确率
        """
        model = model or self.model
        model.eval()
        x, y = self.build_dataset(test_sample_num)
        with torch.no_grad():
            logits = model(x)
            y_pred = torch.argmax(logits, dim=1)
            accuracy = (y_pred == y).float().mean().item()
        return accuracy

    def train(
            self,
            epoch_num: int = 20,
            batch_size: int = 20,
            train_sample: int = 5000,
            custom_loss_fn: Optional[_Loss] = None
    ) -> List[Tuple[float, float]]:
        """
        训练流程，支持自定义损失函数

        :param epoch_num: 训练轮数
        :param batch_size: 批大小
        :param train_sample: 总训练样本数
        :param custom_loss_fn: 自定义损失函数（如果为 None，则使用默认 CrossEntropyLoss）
        :return: 日志 [(loss, acc)]
        """
        train_x, train_y = self.build_dataset(train_sample)
        log: List[Tuple[float, float]] = []
        custom_loss_fn = custom_loss_fn or nn.CrossEntropyLoss()  # 默认 CrossEntropyLoss

        for epoch in range(epoch_num):
            self.model.train()
            watch_loss: List[float] = []
            for batch in range(0, len(train_x), batch_size):
                x_batch = train_x[batch: batch + batch_size]
                y_batch = train_y[batch: batch + batch_size]

                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = custom_loss_fn(logits, y_batch)  # 使用传入的 loss 函数
                loss.backward()
                self.optimizer.step()

                watch_loss.append(loss.item())

            avg_loss = np.mean(watch_loss)
            acc = self.evaluate()
            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
            log.append((avg_loss, acc))

        self.plot_metrics(log)
        return log

    def plot_metrics(self, log: List[Tuple[float, float]]) -> None:
        """
        绘制损失和准确率曲线

        :param log: 训练日志 [(loss, acc)]
        """
        losses, accs = zip(*log)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses, label="Loss")
        plt.title("Training Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(accs, label="Accuracy")
        plt.title("Validation Accuracy")
        plt.legend()
        plt.show()

    def predict(self, input_vecs: List[List[float]]) -> List[int]:
        """
        使用训练好的模型进行预测

        :param input_vecs: 输入样本列表
        :return: 预测类别列表
        """
        self.model.eval()
        x = torch.FloatTensor(input_vecs)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
        for vec, pred, prob in zip(input_vecs, preds, probs):
            print(f"输入: {vec} → 预测类别: {pred.item()}, 概率分布: {prob.numpy()}")
        return preds.tolist()

    def save_model(self, path: str = "model.pth") -> None:
        """
        保存模型

        :param path: 模型保存路径
        """
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存至 {path}")

    def load_model(self, path: str = "model.pth") -> None:
        """
        加载模型

        :param path: 模型加载路径
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"模型已从 {path} 加载")