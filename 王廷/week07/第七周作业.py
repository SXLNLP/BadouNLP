import torch
import numpy as np
import pandas as pd
import concurrent.futures
import gc
import os
import time
import re
import json
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch import nn, optim
from tqdm.auto import tqdm

class TextClassificationExperiment:
    def __init__(self, data_path, output_dir="results"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = self.detect_device()
        self.results = []
        self.config = {
            "MAX_SEQ_LENGTH": 128,
            "BATCH_SIZE": 16,
            "EPOCHS": 3,
            "LEARNING_RATES": [2e-5, 1e-3],
            "HIDDEN_SIZES": [128, 256],
            "MODELS": ['BERT', 'LSTM', 'GRU', 'TextCNN'],
            "FILTER_SIZES": [2, 3, 4],
            "DROPOUT": 0.5,
            "EMBEDDING_DIM": 128,
            "VOCAB_SIZE": 20000
        }
        print(f"实验初始化完成 | 设备: {self.device} | 输出目录: {self.output_dir}")
    
    def detect_device(self):
        """自动检测并选择最佳计算设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def load_data(self):
        """加载并预处理数据集"""
        print("加载数据...")
        try:
            self.data = pd.read_csv(r"D:\BaiduNetdiskDownload\第七周 文本分类\作业\文本分类练习.csv")
            print(f"数据集加载成功: {len(self.data)}条样本")
            
            # 划分数据集
            self.train_texts, self.val_texts, self.train_labels, self.val_labels = train_test_split(
                self.data['review'].tolist(), 
                self.data['label'].tolist(), 
                test_size=0.1, 
                random_state=42
            )
            
            # 分析数据集
            self.analyze_dataset()
            return True
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            return False
    
    def analyze_dataset(self):
        """分析数据集统计信息"""
        self.positive_samples = sum(self.train_labels)
        self.negative_samples = len(self.train_labels) - self.positive_samples
        text_lengths = [len(text) for text in self.train_texts]
        self.average_text_length = np.mean(text_lengths)
        
        print("\n===== 数据集分析 =====")
        print(f"训练集大小: {len(self.train_texts)}")
        print(f"验证集大小: {len(self.val_texts)}")
        print(f"正样本数: {self.positive_samples}")
        print(f"负样本数: {self.negative_samples}")
        print(f"文本平均长度: {self.average_text_length:.2f}字符")
        
        # 保存数据集分析结果
        dataset_info = {
            "训练集大小": len(self.train_texts),
            "验证集大小": len(self.val_texts),
            "正样本数": self.positive_samples,
            "负样本数": self.negative_samples,
            "文本平均长度": round(self.average_text_length, 2)
        }
        with open(self.output_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=4)
    
    def preprocess_data(self):
        """统一预处理数据"""
        print("\n===== 数据预处理 =====")
        
        # BERT数据处理
        self.bert_train_data = self.process_bert_data(self.train_texts, self.train_labels)
        self.bert_val_data = self.process_bert_data(self.val_texts, self.val_labels)
        
        # 非BERT数据处理
        self.non_bert_data = self.process_non_bert_data(self.train_texts + self.val_texts, 
                                                        self.train_labels + self.val_labels)
        
        # 创建数据加载器
        self.bert_train_loader = self.create_dataloader(self.bert_train_data, 'random')
        self.bert_val_loader = self.create_dataloader(self.bert_val_data, 'sequential')
        self.non_bert_train_loader = self.create_dataloader(
            {"input_ids": self.non_bert_data["input_ids"][:len(self.train_texts)], 
             "labels": self.non_bert_data["labels"][:len(self.train_labels)]}, 'random')
        self.non_bert_val_loader = self.create_dataloader(
            {"input_ids": self.non_bert_data["input_ids"][len(self.train_texts):], 
             "labels": self.non_bert_data["labels"][len(self.train_labels):]}, 'sequential')
        
        print("数据预处理完成")
    
    def process_bert_data(self, texts, labels):
        """处理BERT模型数据"""
        tokenizer = BertTokenizer.from_pretrained(r"D:\BaiduNetdiskDownload\第六周 语言模型\bert-base-chinese", return_dict=False)
        input_ids = []
        attention_masks = []
        
        for text in tqdm(texts, desc="BERT数据预处理"):
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.config["MAX_SEQ_LENGTH"],
                padding='max_length',
                truncation=True,
                return_attention_mask=True
            )
            input_ids.append(encoding['input_ids'])
            attention_masks.append(encoding['attention_mask'])
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_masks": torch.tensor(attention_masks),
            "labels": torch.tensor(labels)
        }
    
    def simple_tokenizer(self, text):
        """简单的中文分词器（按字符分词）"""
        # 中文按字符分词
        if re.search("[\u4e00-\u9fff]", text):
            return list(text)
        # 英文按空格分词
        return text.split()
    
    def build_vocab(self, texts):
        """构建自定义词汇表"""
        word_counts = Counter()
        for text in tqdm(texts, desc="构建词汇表"):
            tokens = self.simple_tokenizer(text)
            word_counts.update(tokens)
        
        # 选择最常见的词
        vocab = {'<pad>': 0, '<unk>': 1}
        for word, _ in word_counts.most_common(self.config["VOCAB_SIZE"] - 2):
            vocab[word] = len(vocab)
        
        return vocab
    
    def text_to_sequence(self, text, vocab):
        """将文本转换为索引序列"""
        tokens = self.simple_tokenizer(text)
        return [vocab.get(token, vocab['<unk>']) for token in tokens]
    
    def process_non_bert_data(self, texts, labels):
        """处理非BERT模型数据"""
        # 构建词汇表
        vocab = self.build_vocab(texts)
        
        # 转换为索引序列
        processed_texts = []
        for text in tqdm(texts, desc="非BERT数据预处理"):
            sequence = self.text_to_sequence(text, vocab)
            processed_texts.append(sequence)
        
        # 填充序列
        padded_texts = torch.zeros((len(processed_texts), self.config["MAX_SEQ_LENGTH"]), dtype=torch.long)
        
        for i, seq in enumerate(processed_texts):
            length = min(len(seq), self.config["MAX_SEQ_LENGTH"])
            padded_texts[i, :length] = torch.tensor(seq[:length], dtype=torch.long)
        
        return {
            "input_ids": padded_texts,
            "labels": torch.tensor(labels),
            "vocab": vocab
        }
    
    def create_dataloader(self, data, sampler_type='random'):
        """创建数据加载器"""
        if 'attention_masks' in data:  # BERT数据
            dataset = TensorDataset(data['input_ids'], data['attention_masks'], data['labels'])
        else:  # 非BERT数据
            dataset = TensorDataset(data['input_ids'], data['labels'])
        
        sampler = RandomSampler(dataset) if sampler_type == 'random' else SequentialSampler(dataset)
        return DataLoader(
            dataset, 
            sampler=sampler, 
            batch_size=self.config["BATCH_SIZE"],
            pin_memory=True if self.device.type == 'cuda' else False
        )
    
    def create_model(self, model_name, hidden_size):
        """根据名称创建模型"""
        if model_name == 'BERT':
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-chinese', 
                num_labels=2,
                hidden_dropout_prob=self.config["DROPOUT"]
            )
        else:
            vocab_size = len(self.non_bert_data['vocab'])
            
            if model_name == 'LSTM':
                model = self.LSTMModel(vocab_size, hidden_size)
            elif model_name == 'GRU':
                model = self.GRUModel(vocab_size, hidden_size)
            elif model_name == 'TextCNN':
                model = self.TextCNN(vocab_size, hidden_size)
        
        return model.to(self.device)
    
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 128)
            self.lstm = nn.LSTM(128, hidden_size, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(2 * hidden_size, 2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            embedded = self.embedding(x)
            _, (hidden, _) = self.lstm(embedded)
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            hidden = self.dropout(hidden)
            return self.fc(hidden)
    
    class GRUModel(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 128)
            self.gru = nn.GRU(128, hidden_size, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(2 * hidden_size, 2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            embedded = self.embedding(x)
            _, hidden = self.gru(embedded)
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            hidden = self.dropout(hidden)
            return self.fc(hidden)
    
    class TextCNN(nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 128)
            self.convs = nn.ModuleList([
                nn.Conv2d(1, hidden_size, (k, 128)) for k in [2, 3, 4]  # 固定卷积核大小
            ])
            self.fc = nn.Linear(len([2, 3, 4]) * hidden_size, 2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            embedded = self.embedding(x).unsqueeze(1)  # 添加通道维度
            conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            pooled = [nn.functional.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in conved]
            cat = self.dropout(torch.cat(pooled, dim=1))
            return self.fc(cat)
    
    def train_model(self, model, train_loader, optimizer, criterion):
        """训练单个模型"""
        model.train()
        total_loss = 0
        total_samples = 0
        start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc="训练", leave=False)
        for batch in progress_bar:
            # 准备数据
            inputs = batch[0].to(self.device, non_blocking=True)
            labels = batch[-1].to(self.device, non_blocking=True)
            
            # BERT需要attention mask
            if isinstance(model, BertForSequenceClassification):
                attention_mask = batch[1].to(self.device, non_blocking=True)
                outputs = model(inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            else:
                logits = model(inputs)
                loss = criterion(logits, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            progress_bar.set_postfix(loss=total_loss/total_samples)
        
        avg_loss = total_loss / total_samples
        training_time = time.time() - start_time
        return avg_loss, training_time
    
    def evaluate_model(self, model, val_loader, criterion):
        """评估模型性能"""
        model.eval()
        total_loss = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        start_time = time.time()
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="评估", leave=False)
            for batch in progress_bar:
                # 准备数据
                inputs = batch[0].to(self.device, non_blocking=True)
                labels = batch[-1].to(self.device, non_blocking=True)
                
                # BERT需要attention mask
                if isinstance(model, BertForSequenceClassification):
                    attention_mask = batch[1].to(self.device, non_blocking=True)
                    outputs = model(inputs, attention_mask=attention_mask)
                    logits = outputs.logits
                else:
                    logits = model(inputs)
                
                # 计算损失
                loss = criterion(logits, labels)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                
                # 收集预测结果
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # 计算指标
        avg_loss = total_loss / total_samples
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        eval_time = time.time() - start_time
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "time": eval_time
        }
    
    def run_single_experiment(self, model_name, lr, hidden_size):
        """运行单个实验配置"""
        result = {
            "Model": model_name,
            "Learning Rate": lr,
            "Hidden Size": hidden_size,
            "Status": "Success"
        }
        
        try:
            # 清理显存
            torch.cuda.empty_cache() if self.device.type == "cuda" else None
            
            # 选择正确的数据加载器
            if model_name == 'BERT':
                train_loader = self.bert_train_loader
                val_loader = self.bert_val_loader
            else:
                train_loader = self.non_bert_train_loader
                val_loader = self.non_bert_val_loader
            
            # 创建模型
            model = self.create_model(model_name, hidden_size)
            
            # 设置优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # 训练模型
            train_losses = []
            total_train_time = 0
            for epoch in range(self.config["EPOCHS"]):
                print(f"\nEpoch {epoch+1}/{self.config['EPOCHS']} - {model_name} (lr={lr}, hidden={hidden_size})")
                avg_loss, epoch_time = self.train_model(model, train_loader, optimizer, criterion)
                total_train_time += epoch_time
                train_losses.append(avg_loss)
                print(f"训练损失: {avg_loss:.4f} | 时间: {epoch_time:.2f}s")
            
            # 评估模型
            eval_results = self.evaluate_model(model, val_loader, criterion)
            
            # 计算预测延迟
            start_time = time.time()
            _ = self.evaluate_model(model, val_loader, criterion)
            avg_pred_time = (time.time() - start_time) / len(val_loader.dataset) * 1000  # ms/sample
            
            # 收集结果
            result.update({
                "Training Loss": np.mean(train_losses),
                "Training Time": total_train_time,
                "Validation Loss": eval_results["loss"],
                "Accuracy": eval_results["accuracy"],
                "Precision": eval_results["precision"],
                "Recall": eval_results["recall"],
                "F1 Score": eval_results["f1"],
                "Prediction Time (ms)": avg_pred_time,
                "Evaluation Time": eval_results["time"]
            })
            
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                result["Status"] = "OOM (显存不足)"
            else:
                result["Status"] = f"错误: {str(e)}"
        except Exception as e:
            result["Status"] = f"错误: {str(e)}"
        finally:
            # 清理资源
            if 'model' in locals():
                del model
            torch.cuda.empty_cache() if self.device.type == "cuda" else None
            gc.collect()
        
        return result
    
    def run_parallel_experiments(self):
        """并行运行所有实验配置"""
        tasks = []
        for model_name in self.config["MODELS"]:
            for lr in self.config["LEARNING_RATES"]:
                for hidden_size in self.config["HIDDEN_SIZES"]:
                    # BERT不需要hidden_size参数
                    if model_name == 'BERT' and hidden_size != self.config["HIDDEN_SIZES"][0]:
                        continue
                    tasks.append((model_name, lr, hidden_size))
        
        print(f"\n===== 开始运行 {len(tasks)} 个实验 =====")
        self.results = []
        
        # 使用线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            futures = {executor.submit(self.run_single_experiment, *task): task for task in tasks}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="实验进度"):
                task = futures[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    print(f"完成: {result['Model']} (lr={result['Learning Rate']}, hidden={result['Hidden Size']}) | "
                          f"准确率: {result.get('Accuracy', 0):.4f} | 状态: {result['Status']}")
                except Exception as e:
                    print(f"任务失败: {str(e)}")
    
    def analyze_results(self):
        """分析实验结果并生成报告"""
        if not self.results:
            print("没有可分析的结果")
            return
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(self.results)
        
        # 保存结果到CSV
        results_csv = self.output_dir / "experiment_results.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"\n结果已保存到: {results_csv}")
        
        # 过滤成功实验
        success_df = results_df[results_df["Status"] == "Success"]
        
        if success_df.empty:
            print("没有成功的实验可分析")
            return
        
        # 找出最佳模型
        best_model_idx = success_df["F1 Score"].idxmax()
        best_model = success_df.loc[best_model_idx]
        
        # 生成文本报告
        self.generate_text_report(success_df, best_model)
        
        return success_df
    
    def generate_text_report(self, results_df, best_model):
        """生成文本格式的实验报告"""
        report_path = self.output_dir / "experiment_report.txt"
        
        with open(report_path, "w") as f:
            f.write("="*60 + "\n")
            f.write("文本分类实验报告\n")
            f.write("="*60 + "\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 数据集信息
            f.write("="*60 + "\n")
            f.write("数据集分析\n")
            f.write("="*60 + "\n")
            f.write(f"训练集大小: {len(self.train_texts)}\n")
            f.write(f"验证集大小: {len(self.val_texts)}\n")
            f.write(f"正样本数: {self.positive_samples}\n")
            f.write(f"负样本数: {self.negative_samples}\n")
            f.write(f"文本平均长度: {self.average_text_length:.2f}字符\n\n")
            
            # 实验配置
            f.write("="*60 + "\n")
            f.write("实验配置\n")
            f.write("="*60 + "\n")
            f.write(f"最大序列长度: {self.config['MAX_SEQ_LENGTH']}\n")
            f.write(f"批大小: {self.config['BATCH_SIZE']}\n")
            f.write(f"训练轮数: {self.config['EPOCHS']}\n")
            f.write(f"学习率: {', '.join(map(str, self.config['LEARNING_RATES']))}\n")
            f.write(f"隐藏层大小: {', '.join(map(str, self.config['HIDDEN_SIZES']))}\n")
            f.write(f"模型: {', '.join(self.config['MODELS'])}\n\n")
            
            # 最佳模型
            f.write("="*60 + "\n")
            f.write("最佳模型结果\n")
            f.write("="*60 + "\n")
            f.write(f"模型: {best_model['Model']}\n")
            f.write(f"学习率: {best_model['Learning Rate']}\n")
            f.write(f"隐藏层大小: {best_model['Hidden Size']}\n")
            f.write(f"准确率: {best_model['Accuracy']:.4f}\n")
            f.write(f"F1分数: {best_model['F1 Score']:.4f}\n")
            f.write(f"训练时间: {best_model['Training Time']:.2f}秒\n")
            f.write(f"预测延迟: {best_model['Prediction Time (ms)']:.4f}毫秒/样本\n\n")
            
            # 完整结果
            f.write("="*60 + "\n")
            f.write("完整实验结果 (前10行)\n")
            f.write("="*60 + "\n")
            f.write(results_df.head(10).to_string(index=False))
            
        print(f"实验报告已生成: {report_path}")
    
    def run_full_experiment(self):
        """运行完整实验流程"""
        if not self.load_data():
            return
        
        self.preprocess_data()
        self.run_parallel_experiments()
        self.analyze_results()
        print("\n===== 实验完成 =====")


if __name__ == "__main__":
    # 配置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # 创建并运行实验
    experiment = TextClassificationExperiment(
        data_path="/mnt/文本分类练习.csv",
        output_dir="文本分类实验结果"
    )
    experiment.run_full_experiment()
