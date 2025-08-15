import torch
import math
import numpy as np
from transformers import BertModel
from transformers import BertTokenizer
import torch.nn as nn
import json
import re
from torch.optim import Adam, SGD
from TorchCRF import CRF
from torch.utils.data import TensorDataset, DataLoader
import logging

# 导入性能优化相关模块
from torch.cuda.amp import autocast, GradScaler
import multiprocessing

bert_path="/Users/evan/Downloads/AINLP/week6 语言模型和预训练/bert-base-chinese"
# 保留一个全局 tokenizer 供数据构建使用
bert_tokenizer=BertTokenizer.from_pretrained(bert_path)

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TorchModel(nn.Module):
    def __init__(self, num_labels , bert_path):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, return_dict=True) #return_dict=True: 返回字典风格的 ModelOutput（可属性访问），可访问  hidden_states 和 pooler_output
        
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略填充位置
 
        self.crf_layer = CRF(num_labels)
        self.use_crf = True  # 启用CRF

    def forward(self, x, attention_mask=None, y=None):
        # 1. BERT编码
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # 2. 分类层得到发射分数
        predict = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]
        
        # 3. 根据是否有目标标签决定返回内容
        if y is not None:
            if self.use_crf:
                # 使用CRF计算损失 - TorchCRF库的调用方式
                mask = attention_mask.bool()  # 使用attention_mask作为有效位置的mask
                # 将-100替换为0，因为CRF不处理-100
                target = y.clone()
                target[target == -100] = 0
                return -self.crf_layer(predict, target, mask=mask).mean()
            else:
                # 使用CrossEntropy损失
                return self.loss_fn(predict.view(-1, predict.shape[-1]), y.view(-1))
        else:
            if self.use_crf:
                # CRF解码，返回最优路径
                mask = attention_mask.bool()
                # 对于TorchCRF库，直接调用CRF对象进行解码
                # 当没有target时，CRF应该返回最优路径
                with torch.no_grad():
                    # 尝试不同的调用方式
                    try:
                        # 方式1：直接调用
                        best_paths = []
                        for i in range(predict.size(0)):
                            # 获取每个样本的有效长度
                            seq_len = mask[i].sum().item()
                            if seq_len > 0:
                                # 使用viterbi算法找最优路径（手动实现或使用库函数）
                                path = torch.argmax(predict[i, :seq_len], dim=-1).tolist()
                                best_paths.append(path)
                            else:
                                best_paths.append([])
                        return best_paths
                    except:
                        # 后备方案：使用argmax
                        return torch.argmax(predict, dim=-1)
            else:
                # 直接返回logits
                return predict


#基于bert和它的tokenizer构建训练数据集
def build_dataset():
    with open("/Users/evan/Downloads/AINLP/week9 序列标注问题/ner/ner_data/schema.json", "r", encoding="utf8") as f:
        label_mapping = json.load(f)

    seq_length=242
    
    def load_data_from_file(file_path, data_type="train"):
        """从文件加载数据"""
        seq_string_windows = []
        seq_dataset = []
        seq_string_windows_label = []
        seq_dataset_label = []
        
        logger.info(f"加载{data_type}数据从: {file_path}")
        with open(file_path, "r", encoding="utf8") as d:
            for line in d:  
                if not line.isspace():   
                    # 收集字符和对应标签
                    seq_string_windows.append(line[0])
                    seq_string_windows_label_re = re.sub(r"[^a-zA-Z0-9-]", "", line[2:])
                    #标签映射
                    seq_string_windows_label.append(label_mapping.get(seq_string_windows_label_re, 0))
                else:
                    # 空行表示一个句子结束，保存当前句子
                    if len(seq_string_windows) > 0:
                        seq_dataset.append(seq_string_windows)
                        seq_dataset_label.append(seq_string_windows_label)
                    # 重置临时变量
                    seq_string_windows = []
                    seq_string_windows_label = []

        # 处理最后一个句子（如果文件不以空行结尾）
        if len(seq_string_windows) > 0:
            seq_dataset.append(seq_string_windows)
            seq_dataset_label.append(seq_string_windows_label)
        
        logger.info(f"{data_type}数据加载完成，共{len(seq_dataset)}个句子")
        return seq_dataset, seq_dataset_label
    
    # 加载训练和测试数据
    train_dataset, train_dataset_label = load_data_from_file("/Users/evan/Downloads/AINLP/week9 序列标注问题/ner/ner_data/train", "训练")
    test_dataset, test_dataset_label = load_data_from_file("/Users/evan/Downloads/AINLP/week9 序列标注问题/ner/ner_data/test", "测试")
    
    def process_data(seq_dataset, seq_dataset_label, data_type="train"):
        """处理数据为tensor格式"""
        input_ids_list = []
        attention_masks_list = []
        labels_list = []

        for seq, labels in zip(seq_dataset, seq_dataset_label):
            # 使用 BERT tokenizer 进行编码
            tokenized = bert_tokenizer(seq, return_tensors='pt', padding='max_length', max_length=242, is_split_into_words=True, truncation=True)
            
            # 从 tokenizer 结果中提取需要的部分
            input_ids_list.append(tokenized["input_ids"].squeeze(0)) 
            attention_masks_list.append(tokenized["attention_mask"].squeeze(0))
            
            # 处理标签：padding 到固定长度，用 -100 填充
            pad_len = seq_length - len(labels)
            if pad_len < 0:
                labels = labels[:seq_length]  # 截断过长的标签
                pad_len = 0
            padded_labels = labels + [-100] * pad_len
            labels_list.append(torch.tensor(padded_labels, dtype=torch.long))
        
        # 将列表转换为张量
        input_ids_tensor = torch.stack(input_ids_list)
        attention_mask_tensor = torch.stack(attention_masks_list)
        labels_tensor = torch.stack(labels_list)
        
        logger.info(f"{data_type}数据处理完成 - 形状: {input_ids_tensor.shape}")
        return input_ids_tensor, attention_mask_tensor, labels_tensor
    
    # 处理训练和测试数据
    train_input_ids, train_attention_mask, train_labels = process_data(train_dataset, train_dataset_label, "训练")
    test_input_ids, test_attention_mask, test_labels = process_data(test_dataset, test_dataset_label, "测试")
    
    # 显示数据统计
    print(f"训练集第一个样本字符: {train_dataset[0][:10]}...")
    print(f"训练集第一个样本标签: {train_dataset_label[0][:10]}...")
    print(f"测试集第一个样本字符: {test_dataset[0][:10]}...")
    print(f"测试集第一个样本标签: {test_dataset_label[0][:10]}...")
    
    return train_input_ids, train_attention_mask, train_labels, test_input_ids, test_attention_mask, test_labels


def train(model, train_dataloader, test_input_ids, test_attention_mask, test_labels, 
          optimizer, device, epochs=10):
    """
    训练函数
    Args:
        model: 训练的模型
        train_dataloader: 训练数据加载器
        test_input_ids, test_attention_mask, test_labels: 测试数据
        optimizer: 优化器
        device: 设备
        epochs: 训练轮数
    Returns:
        train_losses, val_losses, val_accuracies: 训练历史
    """
    train_loss = []
    val_losses = []
    val_accuracies = []
    
    logger.info(f"开始训练，共{epochs}个epoch")
    
    for epoch_idx in range(epochs):
        logger.info("epoch %d begin", epoch_idx)
        model.train()
        epoch_train_loss = 0
        
        # 训练阶段
        for index, (batch_input_ids, batch_attention_mask, batch_labels) in enumerate(train_dataloader):
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            loss = model(batch_input_ids, batch_attention_mask, batch_labels)
            if index % 4 == 0:
                logger.info("batch %d loss: %.6f", index, loss.item())
            train_loss.append(loss.item())           
            epoch_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_loss = model(test_input_ids, test_attention_mask, test_labels)
            
            # 计算验证准确率
            if model.use_crf:
                # CRF解码返回最优路径
                val_predictions_list = model(test_input_ids, test_attention_mask)
                # 将列表转换为tensor，CRF返回的是list of lists
                batch_size, max_len = test_input_ids.shape
                val_predictions = torch.zeros(batch_size, max_len, dtype=torch.long, device=test_input_ids.device)
                for i, pred_seq in enumerate(val_predictions_list):
                    seq_len = min(len(pred_seq), max_len)
                    val_predictions[i, :seq_len] = torch.tensor(pred_seq[:seq_len], dtype=torch.long, device=test_input_ids.device)
            else:
                # 普通分类，使用argmax
                val_logits = model(test_input_ids, test_attention_mask)
                val_predictions = torch.argmax(val_logits, dim=-1)
            
            # 计算准确率：直接在2D张量上操作，避免mask带来的bug
            # 创建有效位置的mask：attention_mask=1 且 labels!=-100
            valid_positions = (test_attention_mask == 1) & (test_labels != -100)
            
            if valid_positions.sum() > 0:
                # 提取有效位置的预测和标签
                valid_preds = val_predictions[valid_positions]
                valid_true = test_labels[valid_positions]
                
                # 计算准确率
                correct = (valid_preds == valid_true).sum().item()
                total = valid_positions.sum().item()
                val_accuracy = correct / total
                
                # DEBUG: 输出详细信息
                logger.info(f"验证详情 - 有效位置数: {total}, 正确预测数: {correct}")
                
                # 计算每个实体类别的准确率
                logger.info("=== 各类别准确率 ===")
                
                # 获取标签映射用于显示标签名
                with open("/Users/evan/Downloads/AINLP/week9 序列标注问题/ner/ner_data/schema.json", "r", encoding="utf8") as f:
                    label_mapping = json.load(f)
                id_to_label = {v: k for k, v in label_mapping.items()}
                
                # 计算每个标签的准确率
                for label_id in range(9):  # 假设有9个标签
                    # 找到真实标签为该类别的位置
                    true_mask = valid_true == label_id
                    if true_mask.sum() > 0:
                        # 在这些位置上的预测
                        pred_for_this_label = valid_preds[true_mask]
                        # 计算准确率
                        correct_for_label = (pred_for_this_label == label_id).sum().item()
                        total_for_label = true_mask.sum().item()
                        accuracy_for_label = correct_for_label / total_for_label
                        
                        label_name = id_to_label.get(label_id, f"Label_{label_id}")
                        logger.info(f"{label_name}(ID:{label_id}): {accuracy_for_label:.4f} ({correct_for_label}/{total_for_label})")
                    else:
                        label_name = id_to_label.get(label_id, f"Label_{label_id}")
                        logger.info(f"{label_name}(ID:{label_id}): N/A (0 samples)")
                
    
                # DEBUG: 显示前10个预测结果（仅第一个epoch）
                if epoch_idx == 0:  # 只在第一个epoch显示
                    logger.info(f"前10个预测: {valid_preds[:10].tolist()}")
                    logger.info(f"前10个真实: {valid_true[:10].tolist()}")
            else:
                val_accuracy = 0.0
                logger.warning("没有找到有效的验证位置！")
            
            val_losses.append(val_loss.item())
            val_accuracies.append(val_accuracy)
            logger.info("epoch %d - train_loss: %.6f, val_loss: %.6f, val_accuracy: %.6f", 
                       epoch_idx, epoch_train_loss/len(train_dataloader), val_loss.item(), val_accuracy)
    
    logger.info("训练完成！")

            # 保存模型参数
    model_save_path = "bert_ner_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'num_labels': 9,
        'bert_path': bert_path
    }, model_save_path)
    logger.info("模型已保存到: %s", model_save_path)
    
    return train_loss, val_losses, val_accuracies


    # 测试模型
def test_model(test_text):
    # 加载模型
    checkpoint = torch.load(model_save_path, map_location=device)
    test_model = TorchModel(num_labels=checkpoint['num_labels'], bert_path=checkpoint['bert_path']).to(device)
    test_model.load_state_dict(checkpoint['model_state_dict'])
    test_model.eval()
    
    # 处理测试文本
    test_tokens = list(test_text)
    tokenized = bert_tokenizer(test_tokens, return_tensors='pt', padding='max_length', 
                             max_length=242, is_split_into_words=True, truncation=True)
    
    test_input_ids = tokenized["input_ids"].to(device)
    test_attention_mask = tokenized["attention_mask"].to(device)
    
    # 预测
    with torch.no_grad():
        if test_model.use_crf:
            # CRF解码返回列表
            pred_lists = test_model(test_input_ids, test_attention_mask)
            # 转换为tensor格式
            predictions = torch.zeros_like(test_input_ids)
            for i, pred_list in enumerate(pred_lists):
                seq_len = min(len(pred_list), test_input_ids.size(1))
                predictions[i, :seq_len] = torch.tensor(pred_list[:seq_len])
        else:
            # 普通模式
            logits = test_model(test_input_ids, test_attention_mask)
            predictions = torch.argmax(logits, dim=-1)
        
    # 获取标签映射
    with open("/Users/evan/Downloads/AINLP/week9 序列标注问题/ner/ner_data/schema.json", "r", encoding="utf8") as f:
        label_mapping = json.load(f)
    
    # 反向映射：从数字到标签
    id_to_label = {v: k for k, v in label_mapping.items()}
    
    # 输出结果
    print(f"测试文本: {test_text}")
    print(f"字符\t预测标签")
    print("-" * 20)
    
    for i, char in enumerate(test_tokens):
        if i < len(predictions[0]) and test_attention_mask[0][i] == 1:  # 只显示有效token
            pred_id = predictions[0][i].item()
            pred_label = id_to_label.get(pred_id, f"UNK({pred_id})")
            print(f"{char}\t{pred_label}")
    return predictions

    

if __name__=="__main__":
    train_input_ids, train_attention_mask, train_labels, test_input_ids, test_attention_mask, test_labels = build_dataset()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model=TorchModel(num_labels=9, bert_path=bert_path).to(device)
    
    # 示例：查看第一个样本的 attention_mask
    print(f"第一个样本的 attention_mask: {train_attention_mask[0]}")
    epochs = 10
    optimizer=Adam(model.parameters(),lr=0.00001) #学习率太快会导致 权重变化太快，预训练的信息丢失
    model.train()

    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    
    # 性能优化配置
    batch_size = 64  # 增加batch size，充分利用内存和计算资源
    num_workers = min(8, multiprocessing.cpu_count())  # 多进程数据加载
    
    # 进一步优化建议（可选）：
    # batch_size = 64  # 如果内存充足，可继续增大
    # torch.backends.cudnn.benchmark = True  # 对于固定输入大小，优化CUDNN
    # 考虑使用torch.compile()编译模型（PyTorch 2.0+）
    
    train_dataloader = DataLoader(train_dataset, 
                           batch_size=batch_size, 
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=True if device.type != 'cpu' else False,  # 加速GPU传输
                           persistent_workers=True if num_workers > 0 else False)  # 保持worker进程
    
    test_dataloader = DataLoader(test_dataset, 
                               batch_size=batch_size, 
                               shuffle=False,  # 不shuffle确保验证数据固定
                               num_workers=0)  # 验证不需要多进程
    
    logger.info(f"优化配置 - batch_size: {batch_size}, num_workers: {num_workers}, device: {device}")
    
    # 将测试数据移动到设备上
    test_input_ids = test_input_ids.to(device)
    test_attention_mask = test_attention_mask.to(device)
    test_labels = test_labels.to(device)
    
    logger.info(f"训练集大小: {len(train_dataset)} 样本")
    logger.info(f"测试集大小: {len(test_dataset)} 样本")
    
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    
    # 开始训练
    train_losses, val_losses, val_accuracies = train(
        model=model,
        train_dataloader=train_dataloader,
        test_input_ids=test_input_ids,
        test_attention_mask=test_attention_mask,
        test_labels=test_labels,
        optimizer=optimizer,
        device=device,
        epochs=epochs
    )


    
    # 测试示例
    test_examples = [
        "张三在北京工作",
        "苹果公司发布新产品",
        "我喜欢吃苹果"
    ]
    
    for test_text in test_examples:
        print("\n" + "="*50)
        test_model(test_text)
