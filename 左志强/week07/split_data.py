import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import re
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# 1. 数据加载与预处理
def load_and_preprocess_data(file_path):
    """加载CSV数据并进行预处理"""
    # 尝试不同的编码方式读取文件
    encodings = ['utf-8', 'gbk', 'latin1', 'ISO-8859-1']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"使用 {encoding} 编码成功读取文件")
            break
        except (UnicodeDecodeError, LookupError):
            continue
    
    if df is None:
        raise ValueError("无法读取文件，尝试的编码均失败")
    
    # 检查必要列是否存在
    required_columns = ['review']
    for col in required_columns:
        if col not in df.columns:
            # 尝试自动检测列名
            possible_columns = ['评论', '内容', 'text', 'comment', 'review', '评论内容']
            for possible in possible_columns:
                if possible in df.columns:
                    df.rename(columns={possible: 'review'}, inplace=True)
                    logger.info(f"已将列 '{possible}' 重命名为 'review'")
                    break
            else:
                raise ValueError(f"CSV文件中缺少review列。请确认列名。实际列名: {df.columns.tolist()}")
    
    # 确保label列存在
    if 'label' not in df.columns:
        df['label'] = ""  # 创建空label列
        logger.info("已创建新的label列用于存储分类结果")
    
    # 文本清洗函数
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'【.*?】', '', text)  # 移除广告标签
        text = re.sub(r'@\w+', '', text)    # 移除@提及
        text = re.sub(r'https?://\S+', '', text)  # 移除URL
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)  # 移除非中文字符/标点
        return text.strip()
    
    # 应用清洗
    df['清洗后文本'] = df['review'].apply(clean_text)
    
    # 移除空文本
    df = df[df['清洗后文本'].str.len() > 0]
    
    logger.info(f"数据加载完成，有效样本数: {len(df)}")
    return df

# 2. 创建自定义数据集
class CommentDataset(Dataset):
    """自定义评论数据集"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

# 3. 加载模型和分词器
def load_model():
    """加载预训练的中文情感分析模型"""
    model_name = 'bert-base-chinese'
    
    # 尝试使用在线模型
    try:
        logger.info("尝试从Hugging Face加载模型...")
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            output_attentions=False,
            output_hidden_states=False
        )
        tokenizer = BertTokenizer.from_pretrained(model_name)
        logger.info("成功从Hugging Face加载模型")
    except Exception as e:
        logger.warning(f"在线加载模型失败: {e}")
        
        # 尝试使用本地缓存
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        logger.info(f"尝试从本地缓存加载模型: {cache_dir}")
        
        try:
            model = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=3,
                output_attentions=False,
                output_hidden_states=False,
                local_files_only=True
            )
            tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
            logger.info("成功从本地缓存加载模型")
        except Exception as e2:
            logger.error(f"本地缓存加载也失败: {e2}")
            raise RuntimeError("无法加载模型，请检查网络连接或本地缓存") from e2
    
    model = model.to(device)
    model.eval()  # 设置为评估模式
    return model, tokenizer

# 4. 预测函数
def predict_sentiment(model, tokenizer, dataloader):
    """批量预测情感"""
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # 获取预测类别和概率
            batch_probs = torch.nn.functional.softmax(logits, dim=1)
            batch_preds = torch.argmax(batch_probs, dim=1)
            
            predictions.extend(batch_preds.cpu().numpy())
            probabilities.extend(batch_probs.cpu().numpy())
    
    return predictions, probabilities

# 5. 主函数
def main():
    # 文件路径 (根据实际路径修改)
    file_path = r"C:\Users\Administrator\Desktop\data\文本分类练习.csv"
    
    # 加载数据
    try:
        df = load_and_preprocess_data(file_path)
    except Exception as e:
        logger.error(f"加载数据时出错: {e}")
        return
    
    # 加载模型
    try:
        model, tokenizer = load_model()
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        logger.error("请尝试以下解决方案:")
        logger.error("1. 检查网络连接，确保可以访问 https://huggingface.co")
        logger.error("2. 手动下载模型并设置缓存路径")
        logger.error("3. 使用其他本地模型")
        return
    
    # 创建数据集和数据加载器
    dataset = CommentDataset(df['清洗后文本'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 进行预测
    predictions, probabilities = predict_sentiment(model, tokenizer, dataloader)
    
    # 映射预测结果到情感标签
    sentiment_map = {0: "负面", 1: "中性", 2: "正面"}
    
    # 将结果保存到label列
    df['label'] = [sentiment_map[pred] for pred in predictions]
    
    # 添加概率信息（可选）
    df['负面概率'] = [round(prob[0], 4) for prob in probabilities]
    df['中性概率'] = [round(prob[1], 4) for prob in probabilities]
    df['正面概率'] = [round(prob[2], 4) for prob in probabilities]
    
    # 保存结果
    output_path = r"C:\Users\Administrator\Desktop\data\文本分类结果.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"结果已保存至: {output_path}")
    
    # 显示结果分布
    logger.info("\n情感分布统计:")
    print(df['label'].value_counts())
    
    # 打印样例
    logger.info("\n分类结果样例:")
    print(df[['review', 'label']].head(10))

if __name__ == "__main__":
    main()
    
