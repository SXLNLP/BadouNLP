# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import jieba
import os


def analyze_data(file_path):
    try:
        # 读取数据
        df = pd.read_csv(file_path)
        print("原始数据前5行:\n", df.head())

        # 确定列名
        text_col = "review"
        label_col = "label"

        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError("CSV文件中缺少需要的列")

        # 添加标签文字说明
        df['label_name'] = df[label_col].apply(lambda x: "好评" if x == 1 else "差评")

        # 基本统计
        print("\n=== 基本统计 ===")
        print("样本总数:", len(df))
        print("标签分布:")
        print(df['label_name'].value_counts())

        # 文本长度分析
        print("\n=== 文本长度分析 ===")
        df['text_len'] = df[text_col].apply(lambda x: len(str(x)))
        df['word_count'] = df[text_col].apply(lambda x: len(jieba.lcut(str(x))))

        print("字符长度统计:")
        print(df['text_len'].describe())

        print("\n分词数量统计:")
        print(df['word_count'].describe())

        # 可视化
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        df['label_name'].value_counts().plot(kind='bar', color=['green', 'red'])
        plt.title('Label Distribution')

        plt.subplot(1, 3, 2)
        df['text_len'].hist(bins=30, color='skyblue')
        plt.title('Text Length (Characters)')

        plt.subplot(1, 3, 3)
        df['word_count'].hist(bins=30, color='orange')
        plt.title('Word Count (After Segmentation)')

        plt.tight_layout()

        # 保存结果
        output_dir = "data_analysis"
        os.makedirs(output_dir, exist_ok=True)

        plt.savefig(f"{output_dir}/analysis_results.png")
        df.to_csv(f"{output_dir}/processed_data.csv", index=False)

        print(f"\n分析结果已保存到 {output_dir} 目录")
        plt.show()

    except Exception as e:
        print(f"错误: {str(e)}")
        print("请检查:")
        print("1. 文件路径是否正确")
        print("2. 文件是否包含'review'和'label'列")
        print("3. 文件是否为UTF-8编码")


if __name__ == "__main__":
    file_path = "D:/BaiduNetdiskDownload/第七周 文本分类/week7 文本分类问题/文本分类练习.csv"
    analyze_data(file_path)
