# -*- coding: utf-8 -*-
"""
測試三元組訓練功能
"""

import torch
from config import Config
from model import SiameseNetwork
from loader import DataGenerator

def test_triplet_training():
    print("測試三元組訓練功能...")
    
    # 創建模型
    model = SiameseNetwork(Config)
    print("模型創建成功")
    
    # 創建數據生成器
    dg = DataGenerator(Config["train_data_path"], Config)
    print("數據生成器創建成功")
    
    # 測試三元組樣本生成
    print("\n測試三元組樣本生成:")
    for i in range(3):
        triplet = dg.triplet_training_sample()
        print(f"三元組 {i+1}: anchor={triplet[0].shape}, positive={triplet[1].shape}, negative={triplet[2].shape}")
    
    # 測試模型前向傳播
    print("\n測試模型前向傳播:")
    anchor, positive, negative = dg.triplet_training_sample()
    
    # 添加批次維度
    anchor = anchor.unsqueeze(0)
    positive = positive.unsqueeze(0)
    negative = negative.unsqueeze(0)
    
    loss = model(anchor, positive, negative)
    print(f"三元組損失: {loss.item():.4f}")
    
    # 測試單個句子編碼
    print("\n測試單個句子編碼:")
    single_vector = model(anchor)
    print(f"單個句子向量形狀: {single_vector.shape}")
    
    print("\n三元組訓練功能測試完成！")

if __name__ == "__main__":
    test_triplet_training() 