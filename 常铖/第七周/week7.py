import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# 'review' 列包含评论文本，'sentiment' 列包含标签 (1 表示正面，0 表示负面)
data = pd.read_csv("文本分类练习.csv")

# 查看数据前5行
print(data.head())

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    data['review'], 
    data['sentiment'], 
    test_size=0.2, 
    random_state=42
)

# 特征提取：将文本转换为词频矩阵
vectorizer = CountVectorizer(stop_words='english')
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# 训练朴素贝叶斯分类器
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# 预测
y_pred = classifier.predict(X_test_counts)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
