# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config_homework import Config
from model_homwork import SiameseNetwork, choose_optimizer
from evaluate_homework import Evaluator
from loader_homework import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(config):
    if not os.path.isdir(config["model_path"]):
        os.makedirs(config["model_path"])
    
    train_data = load_data(config["train_data_path"], config, shuffle=True)
    valid_data = load_data(config["valid_data_path"], config, shuffle=False)
    
    model = SiameseNetwork(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("Using GPU")
        model = model.cuda()
    
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)
    
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"Epoch {epoch} begin")
        train_loss = []
        
        for batch_data in train_data:
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            anchor, positive, negative = batch_data
            loss = model(anchor, positive, negative)
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
        
        avg_loss = np.mean(train_loss)
        logger.info(f"Epoch average loss: {avg_loss}")
        evaluator.eval(epoch)
        
        model_path = os.path.join(config["model_path"], f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    main(Config)
