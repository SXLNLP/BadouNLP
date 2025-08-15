# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import logging
from config import Config
from model import NERModel
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = load_data(config["train_data_path"], config)
    model = NERModel().get_model()

    # LoRA 配置
    tuning_tactics = config["tuning_tactics"]
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
    model = get_peft_model(model, peft_config)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    evaluator = Evaluator(config, model, logger)

    for epoch in range(config["epoch"]):
        model.train()
        logger.info("epoch %d begin" % (epoch + 1))
        train_loss = []
        for index, (input_ids, labels) in enumerate(train_data):
            if cuda_flag:
                input_ids, labels = input_ids.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = nn.CrossEntropyLoss()(outputs.logits.view(-1, config["class_num"]), labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)

        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)

if __name__ == "__main__":
    main(Config)
