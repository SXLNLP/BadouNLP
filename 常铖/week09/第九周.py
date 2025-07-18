import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForTokenClassification, AdamW
from tqdm import tqdm

# 假设我们有一些标注好的句子和对应的标签
sentences = ["John lives in New York", "Alice went to Paris"]
labels = [["B-PER", "O", "O", "B-LOC"], ["B-PER", "O", "O", "B-LOC"]]

# 标签到ID的映射
label_to_id = {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3, "I-LOC": 4}

class NERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, label_to_id, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        # Tokenize the sentence
        encoding = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        
        # Encode the labels
        encoded_labels = []
        for item in label:
            if item == 'O':
                encoded_labels.append(0)
            else:
                encoded_labels.append(self.label_to_id[item])
        
        # Pad or truncate the labels
        while len(encoded_labels) < self.max_length:
            encoded_labels.append(0)
        encoded_labels = encoded_labels[:self.max_length]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(encoded_labels)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = NERDataset(sentences, labels, tokenizer, label_to_id)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_id))

optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

for epoch in range(3):  # 训练3个epoch
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

print("Training complete.")
