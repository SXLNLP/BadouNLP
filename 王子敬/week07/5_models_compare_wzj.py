import os
import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import model_dict

import time
import torch
import logging
import random
import numpy as np
import pandas as pd
from config import Config
from loader import load_data, build_vocab
from evaluate import evaluate
from transformers import BertTokenizer
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_and_evaluate(model_type, config, vocab):
    config = config.copy()
    config["model_type"] = model_type
    config["vocab_size"] = len(vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    train_loader, dev_loader = load_data(config, vocab)
    model = model_dict[model_type](config).to(device)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

    start_time = time.time()
    for epoch in range(config["epoch"]):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            loss = model(input_ids, attention_mask, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, dev_loader, device)
        logging.info(f"Model: {model_type} | Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

    final_acc = evaluate(model, dev_loader, device)
    elapsed_time = time.time() - start_time
    logging.info(f"Model {model_type} final accuracy: {final_acc:.4f}, time: {elapsed_time:.2f}s")

    return {
        "Model": model_type,
        "Learning_Rate": config["learning_rate"],
        "Hidden_Size": config["hidden_size"],
        "Batch_Size": config["batch_size"],
        "Max_Length": config["max_length"],
        "Accuracy": final_acc,
        "Time": elapsed_time
    }

if __name__ == '__main__':
    set_seed(Config["seed"])
    vocab = build_vocab(Config)

    all_results = []
    for model_name in ["bert", "lstm", "textcnn", "gatedcnn", "textrcnn"]:
        result = train_and_evaluate(model_name, Config, vocab)
        all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_csv("results.csv", index=False)
    print(df)

