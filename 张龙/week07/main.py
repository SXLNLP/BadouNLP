import os.path
from idlelib.iomenu import encoding

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from model import BertModelClass, LSTMModelClass, RNNModelClass
from config import Config
from loader import load_data
from evaluate import Evaluate
from torch.optim import AdamW
from tqdm import tqdm
from model import get_model

class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text=self.texts[item]
        label=self.labels[item]
        encoding= self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids= encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
    # 返回模型所需的输入
        return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model,data_loader,optimizer,device):
    model.train()
    training_loss = 0

    for batch in tqdm(data_loader,desc='Epoch'):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss=torch.nn.functional.cross_entropy(outputs,labels)

        loss.backward()
        optimizer.step()

        training_loss += loss.item()
    return training_loss/len(data_loader)

def eval_model(model,data_loader,device):
    model.eval()
    y_true=[]
    y_pred=[]
    with torch.no_grad():
        for batch in tqdm(data_loader,desc='Evaluation'):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask)
            predictions = outputs.argmax(dim=-1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    return y_true,y_pred

def main():
    config = Config()
    x_train,x_test,y_train,y_test=load_data(config.data_path,config.test_size)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    vocab_size = len(tokenizer.vocab)

    # 根据配置选择模型

    model = get_model(config, vocab_size)  # 根据配置加载模型
    if config.load_pretrained_model:  # 如果加载预训练模型
        model.load_state_dict(torch.load(config.saved_model_path))
        print(f"Loaded model from {config.saved_model_path}")
    model.to(config.device)  # 移动模型到设备（CPU / GPU）
    train_dataset = ClassificationDataset(x_train,y_train,tokenizer,config.max_seq_length)
    eval_dataset = ClassificationDataset(x_test,y_test,tokenizer,config.max_seq_length)
    train_dataloader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
    eval_dataloader = DataLoader(eval_dataset,batch_size=config.batch_size,shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.num_epochs):
        if not config.load_pretrained_model:
           print(f"Epoch {epoch+1}/{config.num_epochs}")
           train_loss= train_epoch(model,train_dataloader,optimizer,config.device)
           print(f"Training loss: {train_loss:.4f}")


        y_true,y_pred=eval_model(model,eval_dataloader,config.device)
        print(f"True labels: {y_true}")
        print(f"Predicted labels: {y_pred}")
        evaluator = Evaluate()
        metrics=evaluator.evaluate(y_true,y_pred)
        evaluator.print_metrics(metrics)

    if not os.path.exists(config.save_model_dir):
        os.makedirs(config.save_model_dir)
    if not config.load_pretrained_model:
        model_path = os.path.join(config.save_model_dir,"text_classification_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"model saved at {model_path}")

if __name__ == '__main__':
    main()










