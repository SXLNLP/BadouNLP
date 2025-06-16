# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/5/22
# @Author      : liuboyuan
# @Description :
from typing import List

import numpy as np

from classifier_trainer import ClassifierTrainer


if __name__ == "__main__":
    # 1️⃣ 定义分类规则（这里是五分类任务）
    classify_cond = lambda x: np.argmax(x)  # 最大值所在维度作为类别（0~4）

    # 2️⃣ 初始化训练器
    trainer = ClassifierTrainer(
        input_size=5,
        classify_cond=classify_cond,
        learning_rate=0.001
    )

    # 3️⃣ 构造数据集并训练
    print("📊 开始构建数据集和训练...")
    log = trainer.train(
        epoch_num=20,
        batch_size=32,
        train_sample=5000
    )

    # 4️⃣ 计算最终模型准确率
    final_acc = trainer.evaluate(test_sample_num=1000)
    print(f"🎯 最终模型准确率为：{final_acc:.4f}")

    # 5️⃣ 绘制 loss 和 accuracy 曲线（train 函数里已调用 plot_metrics）

    # 6️⃣ 保存模型
    trainer.save_model("multi_class_model.pth")

    # 7️⃣ 加载模型
    trainer.load_model("multi_class_model.pth")

    # 8️⃣ 使用模型做预测
    test_samples: List[List[float]] = [
        [0.1, 0.2, 0.6, 0.1, 0.0],  # 第3维最大 → 预期类别 2
        [0.9, 0.0, 0.0, 0.0, 0.1],  # 第0维最大 → 预期类别 0
        [0.2, 0.3, 0.1, 0.4, 0.0],  # 第3维最大 → 预期类别 3
        [0.1, 0.1, 0.1, 0.1, 0.6],  # 第4维最大 → 预期类别 4
    ]
    print("\n🔮 对以下样本进行预测：")
    predictions = trainer.predict(test_samples)

    # 打印预测结果
    for i, (vec, pred) in enumerate(zip(test_samples, predictions)):
        print(f"样本 {i + 1}: 输入 {vec} → 预测类别 {pred}")