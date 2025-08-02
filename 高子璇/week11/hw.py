import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset

# 加载数据
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# 数据预处理
def preprocess_function(examples, tokenizer):
    inputs = examples["content"]
    outputs = examples["title"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 加载数据
data = load_data("/content/drive/MyDrive/Colab Notebooks/sample_data.json")

# 将数据转换为 Hugging Face Dataset 格式
dataset = Dataset.from_dict({"content": [item["content"] for item in data], "title": [item["title"] for item in data]})

# 加载预训练模型和分词器
model_name = "t5-small"  # 选择合适的预训练模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 预处理数据
tokenized_datasets = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

# 设置训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    do_eval=True,  # 启用评估
    learning_rate=5.6e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
)

# 创建训练器
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()
