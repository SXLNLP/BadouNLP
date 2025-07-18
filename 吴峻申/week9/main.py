import torch
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from transformers import BertConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):
    if not os.path.exists(config["model_path"]):
        os.makedirs(config["model_path"])

    train_data = load_data(config["train_data_path"], config)

    # 创建模型配置
    bert_config = BertConfig.from_pretrained(config["bert_path"])
    bert_config.class_num = config["class_num"]
    bert_config.use_crf = config["use_crf"]

    # 初始化模型
    model = TorchModel(bert_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        logger.info("Using GPU for training")

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"Epoch {epoch} begin")
        train_loss = []

        for batch_idx, batch in enumerate(train_data):
            optimizer.zero_grad()
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if (batch_idx + 1) % 100 == 0:
                logger.info(f"Batch {batch_idx + 1}/{len(train_data)} loss: {np.mean(train_loss[-100:]):.4f}")

        logger.info(f"Epoch average loss: {np.mean(train_loss):.4f}")
        evaluator.eval(epoch)

        model_path = os.path.join(config["model_path"], f"epoch_{epoch}.bin")
        torch.save(model.state_dict(), model_path)

    return model, train_data


if __name__ == "__main__":
    model, train_data = main(Config)
