import pandas as pd
from sklearn.model_selection import train_test_split




def load_data(file_path,test_size=0.20):
    data=pd.read_csv(file_path)
    print(data.head())
    positive_samples = data[data['label'] == 1].shape[0]  # 正样本数
    negative_samples = data[data['label'] == 0].shape[0]  # 负样本数

    # 计算文本的平均长度
    text_lengths = data['review'].apply(len)
    avg_text_length = text_lengths.mean()  # 求平均长度

    # 打印数据分析结果
    print(f"正样本数: {positive_samples}")
    print(f"负样本数: {negative_samples}")
    print(f"文本平均长度: {avg_text_length:.2f} 词")
    texts=data['review'].tolist()
    labels=data['label'].tolist()

    x_train, x_test, y_train, y_test=train_test_split(texts,labels,test_size=test_size,random_state=42)
    return x_train, x_test, y_train, y_test;

