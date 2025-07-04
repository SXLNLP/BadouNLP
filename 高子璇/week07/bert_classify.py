import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# 1. 数据准备
file_path = r'C:\Users\e0973783\Desktop\大模型学习\week7 文本分类问题\week7 文本分类问题\文本分类练习.csv'
df = pd.read_csv(file_path, header=None, names=['label', 'text'])
texts = df['text'].tolist()[1:]
labels = [int(label) for label in df['label'].tolist()[1:]]  # 确保标签为整数

# 加载预训练的 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(r'C:\Users\e0973783\Desktop\大模型学习\bert-base-chinese\bert-base-chinese')


# 编码文本
def encode_texts(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    return encodings, labels


train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

train_encodings, train_labels = encode_texts(train_texts, train_labels)
test_encodings, test_labels = encode_texts(test_texts, test_labels)


# 2. 定义Dataset
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

# 3. 初始化BERT模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 4. 训练配置
training_args = TrainingArguments(
    output_dir='./results',
    do_eval=True,
    eval_steps=500,  # 每多少步评估一次
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 5. 训练模型
trainer.train()

# 6. 测试模型性能
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# 7. 预测耗时测试
import time

model.eval()
with torch.no_grad():
    start_time = time.time()
    _ = trainer.predict(test_dataset)  # 预热
    elapsed_time = time.time() - start_time

    start_time = time.time()
    predictions = trainer.predict(test_dataset)
    elapsed_time = time.time() - start_time

    print(f"\n预测{len(test_dataset)}条文本耗时: {elapsed_time:.4f}秒")
    print(f"平均每条耗时: {(elapsed_time * 1000) / len(test_dataset):.3f}毫秒")
