import os
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 支持的任务类型
TASK_TYPES = ["classification", "generation", "ner", "qa"]


# 新闻数据SFT训练器
class NewsSFTTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.dataset = None

    def load_data(self) -> None:
        """加载并预处理新闻数据集"""
        logger.info(f"Loading dataset from {self.args.data_path}")

        # 从Hugging Face加载或本地加载
        if self.args.dataset_name:
            self.dataset = load_dataset(self.args.dataset_name)
        else:
            # 假设本地数据是JSONL格式
            data_files = {
                "train": os.path.join(self.args.data_path, "train.jsonl"),
                "validation": os.path.join(self.args.data_path, "validation.jsonl"),
                "test": os.path.join(self.args.data_path, "test.jsonl"),
            }
            self.dataset = load_dataset("json", data_files=data_files)

        logger.info(f"Dataset loaded: {self.dataset}")

        # 数据探索与统计
        if self.args.verbose:
            self._explore_data()

    def _explore_data(self) -> None:
        """数据探索与统计"""
        logger.info("Data exploration:")

        # 打印样本数
        for split in self.dataset:
            logger.info(f"  {split} samples: {len(self.dataset[split])}")

        # 打印样本示例
        sample = self.dataset["train"][0]
        logger.info(f"  Sample structure: {list(sample.keys())}")
        logger.info(f"  Sample text: {sample.get('text', sample.get('content', 'N/A'))[:200]}...")

        # 计算平均长度（如果有text字段）
        if "text" in self.dataset["train"].features:
            lengths = [len(self.tokenizer(text)["input_ids"]) for text in tqdm(
                self.dataset["train"]["text"][:1000], desc="Calculating text lengths"
            )]
            logger.info(f"  Average text length: {np.mean(lengths):.1f} tokens")
            logger.info(f"  Max text length: {np.max(lengths)} tokens")

    def load_model_and_tokenizer(self) -> None:
        """加载预训练模型和分词器"""
        logger.info(f"Loading model {self.args.model_name} and tokenizer")

        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
            padding_side="right",
        )

        # 特殊token处理
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        if self.args.use_4bit:
            # 4位量化加载
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # 普通加载
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
            )

        logger.info(f"Model loaded: {self.args.model_name}")

    def preprocess_data(self) -> None:
        """预处理数据集"""
        logger.info(f"Preprocessing data for task: {self.args.task_type}")

        # 根据任务类型选择预处理函数
        if self.args.task_type == "classification":
            preprocess_fn = self._preprocess_classification_data
        elif self.args.task_type == "generation":
            preprocess_fn = self._preprocess_generation_data
        elif self.args.task_type == "ner":
            preprocess_fn = self._preprocess_ner_data
        elif self.args.task_type == "qa":
            preprocess_fn = self._preprocess_qa_data
        else:
            raise ValueError(f"Unsupported task type: {self.args.task_type}")

        # 应用预处理
        self.dataset = self.dataset.map(
            preprocess_fn,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
            desc="Preprocessing dataset",
        )

        logger.info("Data preprocessing completed")

    def _preprocess_classification_data(self, examples: Dict[str, List]) -> Dict[str, List]:
        """预处理分类任务数据"""
        # 构造prompt模板
        if self.args.prompt_template:
            template = self.args.prompt_template
        else:
            # 默认分类模板
            template = "分类新闻主题：\n{text}\n候选类别：{categories}\n正确类别：{label}"

        # 获取类别列表（如果有）
        categories = getattr(self.args, "categories", [])
        category_str = ", ".join(categories) if categories else ""

        # 构造prompts
        prompts = []
        for i in range(len(examples["text"])):
            text = examples["text"][i]
            label = examples["label"][i]

            # 替换模板变量
            prompt = template.format(
                text=text,
                categories=category_str,
                label=label
            )
            prompts.append(prompt)

        # 编码
        model_inputs = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.args.max_length,
            padding="max_length" if self.args.pad_to_max_length else False,
        )

        # 构造标签（分类任务直接使用label ID）
        if "label" in examples:
            model_inputs["labels"] = examples["label"]

        return model_inputs

    def _preprocess_generation_data(self, examples: Dict[str, List]) -> Dict[str, List]:
        """预处理生成任务数据"""
        # 构造指令和输入文本
        instructions = examples.get("instruction", ["生成摘要"] * len(examples["text"]))
        inputs = examples["text"]
        outputs = examples["output"]

        # 构造完整prompt
        prompts = []
        for instr, text in zip(instructions, inputs):
            prompt = f"{instr}\n\n{text}\n\n答案："
            prompts.append(prompt)

        # 编码输入
        model_inputs = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.args.max_length,
            padding="max_length" if self.args.pad_to_max_length else False,
        )

        # 编码输出并构造标签（忽略输入部分的loss）
        labels = self.tokenizer(
            outputs,
            truncation=True,
            max_length=self.args.max_length,
            padding="max_length" if self.args.pad_to_max_length else False,
        )["input_ids"]

        # 构造最终标签（-100表示忽略计算loss的位置）
        model_inputs["labels"] = []
        for i in range(len(model_inputs["input_ids"])):
            input_len = len(model_inputs["input_ids"][i])
            output_len = len(labels[i])

            # 如果总长度超过max_length，截断输出
            if input_len + output_len > self.args.max_length:
                output_len = self.args.max_length - input_len

            # 完整标签 = 输入部分(-100) + 输出部分
            full_labels = [-100] * input_len + labels[i][:output_len]

            # 如果需要填充到max_length
            if self.args.pad_to_max_length and len(full_labels) < self.args.max_length:
                full_labels += [-100] * (self.args.max_length - len(full_labels))

            model_inputs["labels"].append(full_labels)

        return model_inputs

    def _preprocess_ner_data(self, examples: Dict[str, List]) -> Dict[str, List]:
        """预处理命名实体识别数据"""
        # 实现NER数据预处理逻辑
        # 这里简化处理，实际应根据具体标注格式调整
        raise NotImplementedError("NER data preprocessing not implemented yet")

    def _preprocess_qa_data(self, examples: Dict[str, List]) -> Dict[str, List]:
        """预处理问答数据"""
        # 实现问答数据预处理逻辑
        # 构造[问题 + 上下文 -> 答案]的格式
        raise NotImplementedError("QA data preprocessing not implemented yet")

    def setup_lora(self) -> None:
        """配置LoRA参数高效微调"""
        logger.info(f"Setting up LoRA with r={self.args.lora_r}, alpha={self.args.lora_alpha}")

        # 配置LoRA
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            target_modules=self.args.lora_target_modules,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)

        # 打印可训练参数
        logger.info("Trainable parameters:")
        self.model.print_trainable_parameters()

    def train(self) -> None:
        """训练模型"""
        logger.info(f"Starting training with args: {self.args}")

        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            learning_rate=self.args.learning_rate,
            per_device_train_batch_size=self.args.train_batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            warmup_ratio=self.args.warmup_ratio,
            weight_decay=self.args.weight_decay,
            lr_scheduler_type=self.args.lr_scheduler_type,
            num_train_epochs=self.args.num_train_epochs,
            logging_dir=f"{self.args.output_dir}/logs",
            logging_steps=self.args.logging_steps,
            save_strategy=self.args.save_strategy,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            evaluation_strategy=self.args.evaluation_strategy,
            eval_steps=self.args.eval_steps,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            tf32=self.args.tf32,
            dataloader_num_workers=self.args.dataloader_num_workers,
            load_best_model_at_end=self.args.load_best_model_at_end,
            metric_for_best_model=self.args.metric_for_best_model,
            greater_is_better=self.args.greater_is_better,
            report_to=self.args.report_to,
            remove_unused_columns=False,
        )

        # 配置数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # 配置评估指标
        compute_metrics = self._get_compute_metrics() if self.args.task_type == "classification" else None

        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # 开始训练
        train_result = trainer.train()

        # 保存模型
        logger.info(f"Saving model to {self.args.output_dir}")
        trainer.save_model()

        # 保存训练结果
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def _get_compute_metrics(self):
        """获取评估指标计算函数（分类任务）"""

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            # 对于分类任务，取logits的argmax作为预测结果
            predictions = np.argmax(predictions, axis=1)

            # 计算准确率和F1
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average="weighted")

            return {"accuracy": accuracy, "f1": f1}

        return compute_metrics

    def evaluate(self) -> None:
        """评估模型"""
        if "test" not in self.dataset:
            logger.warning("No test dataset found. Skipping evaluation.")
            return

        logger.info("Starting evaluation on test dataset")

        # 加载最佳模型
        if os.path.exists(os.path.join(self.args.output_dir, "adapter_model")):
            logger.info(f"Loading best model from {self.args.output_dir}")
            self.model = PeftModel.from_pretrained(
                self.model,
                os.path.join(self.args.output_dir, "adapter_model"),
            )

        # 配置评估参数
        eval_args = TrainingArguments(
            output_dir=self.args.output_dir,
            per_device_eval_batch_size=self.args.eval_batch_size,
            remove_unused_columns=False,
        )

        # 配置数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # 配置评估指标
        compute_metrics = self._get_compute_metrics() if self.args.task_type == "classification" else None

        # 创建评估器
        trainer = Trainer(
            model=self.model,
            args=eval_args,
            eval_dataset=self.dataset["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # 评估
        metrics = trainer.evaluate()

        # 保存评估结果
        logger.info(f"Evaluation metrics: {metrics}")
        with open(os.path.join(self.args.output_dir, "evaluation_results.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def merge_lora_weights(self) -> None:
        """合并LoRA权重到基础模型"""
        if not os.path.exists(os.path.join(self.args.output_dir, "adapter_model")):
            logger.error(f"LoRA adapter not found at {self.args.output_dir}")
            return

        logger.info("Merging LoRA weights into base model")

        # 加载LoRA模型
        model = PeftModel.from_pretrained(
            self.model,
            os.path.join(self.args.output_dir, "adapter_model"),
        )

        # 合并权重
        merged_model = model.merge_and_unload()

        # 保存合并后的模型
        merged_output_dir = os.path.join(self.args.output_dir, "merged_model")
        logger.info(f"Saving merged model to {merged_output_dir}")
        merged_model.save_pretrained(merged_output_dir)
        self.tokenizer.save_pretrained(merged_output_dir)

        logger.info("LoRA weights merged successfully")

    def predict(self, text: str) -> str:
        """使用训练好的模型进行预测"""
        # 加载最佳模型
        if os.path.exists(os.path.join(self.args.output_dir, "adapter_model")):
            logger.info(f"Loading best model from {self.args.output_dir}")
            self.model = PeftModel.from_pretrained(
                self.model,
                os.path.join(self.args.output_dir, "adapter_model"),
            )

        # 设置为评估模式
        self.model.eval()

        # 构造输入
        if self.args.task_type == "classification":
            # 分类任务输入
            categories = getattr(self.args, "categories", [])
            category_str = ", ".join(categories) if categories else ""
            prompt = f"分类新闻主题：\n{text}\n候选类别：{category_str}\n正确类别："
        elif self.args.task_type == "generation":
            # 生成任务输入
            instruction = getattr(self.args, "instruction", "生成摘要")
            prompt = f"{instruction}\n\n{text}\n\n答案："
        else:
            # 其他任务类型
            prompt = text

        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 生成输出
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                num_beams=self.args.num_beams,
                do_sample=self.args.do_sample,
            )

        # 解码输出
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取生成部分（去掉输入部分）
        if self.args.task_type == "generation":
            output_text = output_text.replace(prompt, "").strip()

        return output_text


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="新闻数据SFT训练")

    # 数据相关参数
    parser.add_argument("--data_path", type=str, default=None, help="数据路径")
    parser.add_argument("--dataset_name", type=str, default=None, help="Hugging Face数据集名称")
    parser.add_argument("--task_type", type=str, default="classification", choices=TASK_TYPES, help="任务类型")
    parser.add_argument("--prompt_template", type=str, default=None, help="自定义提示模板")
    parser.add_argument("--categories", nargs="+", default=[], help="分类任务的类别列表")
    parser.add_argument("--instruction", type=str, default="生成摘要", help="生成任务的指令")

    # 模型相关参数
    parser.add_argument("--model_name", type=str, default="qwen/Qwen-1.5B-Instruct", help="预训练模型名称")
    parser.add_argument("--use_4bit", action="store_true", help="使用4位量化")
    parser.add_argument("--fp16", action="store_true", help="使用FP16混合精度")
    parser.add_argument("--bf16", action="store_true", help="使用BF16混合精度")
    parser.add_argument("--tf32", action="store_true", help="使用TF32精度")

    # LoRA相关参数
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA缩放系数")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout率")
    parser.add_argument("--lora_target_modules", nargs="+",
                        default=["q_proj", "v_proj"], help="LoRA目标模块")

    # 训练相关参数
    parser.add_argument("--output_dir", type=str, default="./news_sft_output", help="输出目录")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="生成时的最大新token数")
    parser.add_argument("--pad_to_max_length", action="store_true", help="是否填充到最大长度")
    parser.add_argument("--train_batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="评估批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="学习率预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型")

    # 评估和保存相关参数
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=None, help="评估步数")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="保存策略")
    parser.add_argument("--save_steps", type=int, default=None, help="保存步数")
    parser.add_argument("--save_total_limit", type=int, default=3, help="最多保存的检查点数量")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="训练结束后加载最佳模型")
    parser.add_argument("--metric_for_best_model", type=str, default="loss", help="用于选择最佳模型的指标")
    parser.add_argument("--greater_is_better", action="store_true", help="指标值越大越好")

    # 日志和报告相关参数
    parser.add_argument("--logging_steps", type=int, default=100, help="日志记录步数")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="报告工具")
    parser.add_argument("--verbose", action="store_true", help="是否打印详细信息")

    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="top-p采样")
    parser.add_argument("--top_k", type=int, default=40, help="top-k采样")
    parser.add_argument("--num_beams", type=int, default=1, help="束搜索宽度")
    parser.add_argument("--do_sample", action="store_true", help="是否使用采样")

    # 预测参数
    parser.add_argument("--predict_text", type=str, default=None, help="用于预测的文本")

    # 合并参数
    parser.add_argument("--merge_lora", action="store_true", help="是否合并LoRA权重")

    return parser.parse_args()


def main() -> None:
    """主函数"""
    args = parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化训练器
    trainer = NewsSFTTrainer(args)

    # 加载模型和分词器
    trainer.load_model_and_tokenizer()

    # 如果是训练模式
    if args.data_path or args.dataset_name:
        # 加载数据
        trainer.load_data()

        # 预处理数据
        trainer.preprocess_data()

        # 设置LoRA
        trainer.setup_lora()

        # 训练模型
        trainer.train()

        # 评估模型
        trainer.evaluate()

    # 合并LoRA权重
    if args.merge_lora:
        trainer.merge_lora_weights()

    # 预测
    if args.predict_text:
        result = trainer.predict(args.predict_text)
        print(f"预测结果: {result}")


if __name__ == "__main__":
    main()
