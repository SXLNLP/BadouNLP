# -*- coding: utf-8 -*-
import os
import logging
import numpy as np
import torch
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    os.makedirs(config["model_path"], exist_ok=True)
    train_data = load_data(config["train_data_path"], config, shuffle=True)
    valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    model = TorchModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)

    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    for epoch in range(1, config["epoch"] + 1):
        model.train()
        train_loss = []
        for step, batch in enumerate(train_data):
            input_ids, labels = batch
            input_ids = input_ids.to(device)   # 明确移动到 GPU
            labels = labels.to(device)

            # attention_mask 自动从 input_ids 生成，也可以指定为 batch 中传入的值
            attention_mask = (input_ids != 0).long()

            optimizer.zero_grad()
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            if step % max(1, len(train_data) // 2) == 0:
                logger.info("batch loss %f" % loss.item())

        logger.info("epoch %d average loss: %f", epoch, np.mean(train_loss))
        evaluator.eval(epoch)

        torch.save(model.state_dict(),
                os.path.join(config["model_path"], f"epoch_{epoch}.pth"))

if __name__ == "__main__":
    main(Config)
