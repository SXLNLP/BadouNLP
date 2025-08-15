# -*- coding: utf-8 -*-

import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report
from loader import load_data

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model  
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        
        # 获取标签映射
        self.index_to_label = self.valid_data.dataset.index_to_label
        self.label_to_index = self.valid_data.dataset.label_to_index
        
        # 确定设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def extract_entities(self, tokens, labels, index_to_label):
        """从BIO标签序列中提取实体"""
        entities = []
        current_entity = None
        
        for i, (token, label_id) in enumerate(zip(tokens, labels)):
            if label_id == -100:  # 跳过填充token
                continue
                
            label = index_to_label.get(label_id, 'O')
            
            if label == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
            elif label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                entity_type = label[2:]  # 去掉'B-'前缀
                current_entity = {
                    'type': entity_type,
                    'start': i,
                    'end': i,
                    'text': str(token) if hasattr(token, '__str__') else ''
                }
            elif label.startswith('I-'):
                if current_entity and label[2:] == current_entity['type']:
                    current_entity['end'] = i
                    current_entity['text'] += str(token) if hasattr(token, '__str__') else ''
                else:
                    # 不匹配的I标签，当作新实体开始
                    if current_entity:
                        entities.append(current_entity)
                    entity_type = label[2:]
                    current_entity = {
                        'type': entity_type,
                        'start': i,
                        'end': i,
                        'text': str(token) if hasattr(token, '__str__') else ''
                    }
        
        if current_entity:
            entities.append(current_entity)
        
        return entities

    def calculate_entity_metrics(self, true_entities, pred_entities):
        """计算实体级别的precision, recall, F1"""
        # 转换为集合，用于计算交集
        true_set = {(e['type'], e['start'], e['end']) for e in true_entities}
        pred_set = {(e['type'], e['start'], e['end']) for e in pred_entities}
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1, tp, fp, fn

    def calculate_entity_metrics_by_type(self, true_entities, pred_entities):
        """按实体类型计算详细指标"""
        # 按类型分组
        true_by_type = defaultdict(list)
        pred_by_type = defaultdict(list)
        
        for entity in true_entities:
            true_by_type[entity['type']].append((entity['start'], entity['end']))
        
        for entity in pred_entities:
            pred_by_type[entity['type']].append((entity['start'], entity['end']))
        
        # 收集所有实体类型
        all_types = set(true_by_type.keys()) | set(pred_by_type.keys())
        
        type_metrics = {}
        for entity_type in all_types:
            true_set = set(true_by_type[entity_type])
            pred_set = set(pred_by_type[entity_type])
            
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            type_metrics[entity_type] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'support': len(true_set)
            }
        
        return type_metrics

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_true_entities = []
        all_pred_entities = []
        
        with torch.no_grad():
            for index, batch_data in enumerate(self.valid_data):
                batch_data = [d.to(self.device) for d in batch_data]
                input_ids, labels = batch_data
                
                # 创建attention_mask
                attention_mask = (input_ids != 0).float()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred_logits = outputs.logits
                predictions = torch.argmax(pred_logits, dim=-1)
                
                # 处理每个样本
                for i in range(input_ids.size(0)):
                    # 获取有效位置（非填充）
                    valid_positions = (labels[i] != -100)
                    if not valid_positions.any():
                        continue
                    
                    # 提取有效的预测和标签
                    valid_preds = predictions[i][valid_positions].cpu().numpy()
                    valid_labels = labels[i][valid_positions].cpu().numpy()
                    valid_tokens = range(len(valid_preds))  # 简化的token表示
                    
                    all_predictions.extend(valid_preds)
                    all_labels.extend(valid_labels)
                    
                    # 提取实体
                    true_entities = self.extract_entities(valid_tokens, valid_labels, self.index_to_label)
                    pred_entities = self.extract_entities(valid_tokens, valid_preds, self.index_to_label)
                    
                    all_true_entities.extend(true_entities)
                    all_pred_entities.extend(pred_entities)
        
        # 计算token级别准确率
        token_accuracy = sum([p == l for p, l in zip(all_predictions, all_labels)]) / len(all_predictions) if all_predictions else 0.0
        
        # 计算整体实体级别指标
        entity_precision, entity_recall, entity_f1, total_tp, total_fp, total_fn = self.calculate_entity_metrics(all_true_entities, all_pred_entities)
        
        # 计算每个实体类型的详细指标
        type_metrics = self.calculate_entity_metrics_by_type(all_true_entities, all_pred_entities)
        
        # 输出结果
        self.logger.info("="*60)
        self.logger.info(f"📊 第{epoch}轮验证结果")
        self.logger.info("="*60)
        
        # Token级别指标
        self.logger.info(f"🏷️  Token级别准确率: {token_accuracy:.4f}")
        
        # 整体实体级别指标
        self.logger.info(f"🎯 实体级别指标:")
        self.logger.info(f"   精确率(Precision): {entity_precision:.4f}")
        self.logger.info(f"   召回率(Recall): {entity_recall:.4f}")
        self.logger.info(f"   F1分数: {entity_f1:.4f}")
        self.logger.info(f"   TP={total_tp}, FP={total_fp}, FN={total_fn}")
        
        # 各实体类型详细指标
        self.logger.info(f"\n📈 各实体类型表现:")
        self.logger.info(f"{'实体类型':<12} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'支持数':<6} {'TP':<4} {'FP':<4} {'FN':<4}")
        self.logger.info("-" * 70)
        
        # 按F1分数降序排列
        sorted_types = sorted(type_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
        
        for entity_type, metrics in sorted_types:
            self.logger.info(
                f"{entity_type:<12} "
                f"{metrics['precision']:<8.3f} "
                f"{metrics['recall']:<8.3f} "
                f"{metrics['f1']:<8.3f} "
                f"{metrics['support']:<6d} "
                f"{metrics['tp']:<4d} "
                f"{metrics['fp']:<4d} "
                f"{metrics['fn']:<4d}"
            )
        
        # 计算宏平均和微平均
        if type_metrics:
            macro_precision = sum([m['precision'] for m in type_metrics.values()]) / len(type_metrics)
            macro_recall = sum([m['recall'] for m in type_metrics.values()]) / len(type_metrics)
            macro_f1 = sum([m['f1'] for m in type_metrics.values()]) / len(type_metrics)
            
            self.logger.info("-" * 70)
            self.logger.info(f"{'宏平均':<12} {macro_precision:<8.3f} {macro_recall:<8.3f} {macro_f1:<8.3f}")
            self.logger.info(f"{'微平均':<12} {entity_precision:<8.3f} {entity_recall:<8.3f} {entity_f1:<8.3f}")
        
        # 实体统计
        total_true_entities = len(all_true_entities)
        total_pred_entities = len(all_pred_entities)
        self.logger.info(f"\n📊 实体统计:")
        self.logger.info(f"   真实实体总数: {total_true_entities}")
        self.logger.info(f"   预测实体总数: {total_pred_entities}")
        self.logger.info(f"   正确识别实体: {total_tp}")
        
        self.logger.info("="*60)
        
        return entity_f1  # 返回实体F1作为主要评估指标

    def write_stats(self, labels, pred_results):
        # 保留原有接口，但现在在eval方法中直接处理
        pass

    def show_stats(self):
        # 保留原有接口，但现在在eval方法中直接处理
        return 0.0
