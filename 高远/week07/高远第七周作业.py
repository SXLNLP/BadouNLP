import pandas as pd
import jieba
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# 1. 读取数据
df = pd.read_csv('文本分类练习.csv')
print(df.head())

# 假设列名是 'text' 和 'label'
df.columns = df.columns.str.strip()  # 去除列名空格
texts = df['review'].astype(str)
labels = df['label']

# 2. 数据分析
print("总样本数：", len(df))
print("正样本数：", sum(labels == 1))
print("负样本数：", sum(labels == 0))
print("文本平均长度：", texts.apply(len).mean())

# 3. 分词（用于非BERT模型）
texts_cut = texts.apply(lambda x: " ".join(jieba.cut(x)))

# 划分训练集/验证集
X_train, X_test, y_train, y_test = train_test_split(texts_cut, labels, test_size=0.2, random_state=42)

# TF-IDF 向量
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 存储结果
results = []

# 4. 模型1：Naive Bayes
model_nb = MultinomialNB()
start = time.time()
model_nb.fit(X_train_tfidf, y_train)
acc = accuracy_score(y_test, model_nb.predict(X_test_tfidf))
predict_time = time.time() - start
results.append(['NaiveBayes', '-', '-', acc, predict_time])

# 5. 模型2：Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
start = time.time()
model_lr.fit(X_train_tfidf, y_train)
acc = accuracy_score(y_test, model_lr.predict(X_test_tfidf))
predict_time = time.time() - start
results.append(['LogisticRegression', '-', '-', acc, predict_time])

# 6. 模型3：Linear SVM
model_svm = LinearSVC()
start = time.time()
model_svm.fit(X_train_tfidf, y_train)
acc = accuracy_score(y_test, model_svm.predict(X_test_tfidf))
predict_time = time.time() - start
results.append(['SVM', '-', '-', acc, predict_time])


# 7. 模型4：BERT 中文模型（bert-base-chinese）
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
X_train_enc = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
X_test_enc = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

train_dataset = TextDataset(X_train_enc, y_train)
test_dataset = TextDataset(X_test_enc, y_test)

model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# 预测并计算时间
start = time.time()
preds = trainer.predict(test_dataset).predictions
acc = accuracy_score(y_test, preds.argmax(axis=1))
predict_time = time.time() - start
results.append(['BERT', '2e-5', '768', acc, predict_time])

# 8. 输出总结表格
summary_df = pd.DataFrame(results, columns=["Model", "Learning_Rate", "Hidden_Size", "acc", "time(预测100条耗时)"])
print(summary_df)
summary_df.to_csv('模型效果对比.csv', index=False)
