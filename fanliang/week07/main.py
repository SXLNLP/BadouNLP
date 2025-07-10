# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loaderremark import load_data

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 优先使用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        # debug train_data CODE,show the shape of input_ids and label
        # dataset = train_data.dataset  # DataLoader 的底层 Dataset
        # for idx in range(len(dataset)):
        #     input_ids, label = dataset[idx]
        #     if input_ids.shape[0] != config["max_length"]:
        #         print(f"Sample {idx}: input_ids shape: {input_ids.shape}, label shape: {label.shape}")
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels) #进入model之前的都是词索引的tensor，而非词向量的tensor
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)
        
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    # 存储所有实验结果
    results = []
    
    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    for model in ["gated_cnn", 'bert', 'lstm']:
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        accuracy = main(Config)
                        results.append({
                            'model_type': model,
                            'learning_rate': lr,
                            'hidden_size': hidden_size,
                            'batch_size': batch_size,
                            'pooling_style': pooling_style,
                            'accuracy': accuracy
                        })
                        print("最后一轮准确率：", accuracy, "当前配置：", Config)
    
    # 显示结果表格
    print("\n" + "="*80)
    print("实验结果汇总表")
    print("="*80)
    print(f"{'模型类型':<12} {'学习率':<10} {'隐藏层':<8} {'Batch':<8} {'Pooling':<8} {'准确率':<8}")
    print("-"*80)
    
    for result in results:
        print(f"{result['model_type']:<12} {result['learning_rate']:<10} {result['hidden_size']:<8} "
              f"{result['batch_size']:<8} {result['pooling_style']:<8} {result['accuracy']:.4f}")
    
    # 找出最佳配置
    best_result = max(results, key=lambda x: x['accuracy'])
    print("-"*80)
    print(f"最佳配置: {best_result['model_type']}, 学习率: {best_result['learning_rate']}, "
          f"Batch: {best_result['batch_size']}, Pooling: {best_result['pooling_style']}, "
          f"准确率: {best_result['accuracy']:.4f}")
    print("="*80)
    
    # 创建紧凑的图片，让matplotlib自动处理布局
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, len(results)*0.2 + 1.5), 
                                       gridspec_kw={'height_ratios': [0.15, 0.75, 0.1]})
    
    # 标题区域
    ax1.axis('off')
    ax1.text(0.5, 0.5, '实验结果汇总表', fontsize=14, fontweight='bold', 
             ha='center', va='center', transform=ax1.transAxes)
    
    # 表格区域
    ax2.axis('off')
    
    # 准备表格数据
    table_data = []
    headers = ['模型类型', '学习率', '隐藏层', 'Batch', 'Pooling', '准确率']
    
    for result in results:
        table_data.append([
            result['model_type'],
            str(result['learning_rate']),
            str(result['hidden_size']),
            str(result['batch_size']),
            result['pooling_style'],
            f"{result['accuracy']:.4f}"
        ])
    
    # 创建表格
    table = ax2.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    
    # 表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.0)
    
    # 设置表头颜色
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 高亮最佳配置行
    best_idx = max(range(len(results)), key=lambda i: results[i]['accuracy'])
    for i in range(len(headers)):
        table[(best_idx + 1, i)].set_facecolor('#FFD700')
    
    # 结果说明区域
    ax3.axis('off')
    best_result = max(results, key=lambda x: x['accuracy'])
    ax3.text(0.5, 0.5, 
             f'最佳配置: {best_result["model_type"]}, 学习率: {best_result["learning_rate"]}, '
             f'Batch: {best_result["batch_size"]}, Pooling: {best_result["pooling_style"]}, '
             f'准确率: {best_result["accuracy"]:.4f}',
             ha='center', va='center', fontsize=10, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.8),
             transform=ax3.transAxes)
    
    # 让三个子图紧挨着
    plt.tight_layout()
    plt.savefig('experiment_results_table.png', dpi=300, bbox_inches='tight')
    plt.show()


