import random
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Dataset generation
class APositionDataset(Dataset):
    def __init__(self, size, vector_dim=16):
        # vocab and mappings
        self.vocab = list(string.ascii_lowercase)
        self.char2idx = {ch: i+1 for i, ch in enumerate(self.vocab)}
        self.embedding_dim = vector_dim
        
        # generate data
        self.strings = []
        self.labels = []  # multi-hot vector length 5
        for _ in range(size):
            while True:
                s = ''.join(random.choices(self.vocab, k=5))
                if 'a' in s:
                    break
            self.strings.append(s)
            label = [1 if ch=='a' else 0 for ch in s]
            self.labels.append(label)
        
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx):
        s = self.strings[idx]
        indices = [self.char2idx[ch] for ch in s]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float)

# 2. Model definition
class ARNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, seq_len=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size+1, emb_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=emb_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, emb_dim)
        out, _ = self.rnn(emb)   # (batch, seq_len, hidden_dim)
        logits = self.fc(out).squeeze(-1)  # (batch, seq_len)
        return logits

# 3. Training and evaluation function
def train_and_evaluate(dataset_size=1000, vector_dim=16, hidden_dim=32, 
                       batch_size=32, epochs=10, lr=1e-3):
    # prepare dataset
    dataset = APositionDataset(dataset_size, vector_dim)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    # model, loss, optimizer
    model = ARNN(vocab_size=26, emb_dim=vector_dim, hidden_dim=hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # training loop
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
    
    # evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            preds = torch.sigmoid(logits) >= 0.5
            all_preds.append(preds.int().cpu())
            all_targets.append(batch_y.int().cpu())
    preds_tensor = torch.cat(all_preds).view(-1)
    targets_tensor = torch.cat(all_targets).view(-1)
    
    acc = accuracy_score(targets_tensor, preds_tensor)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_tensor, preds_tensor, average='binary', zero_division=0
    )
    print("\nEvaluation on Test Set:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    return model, (acc, precision, recall, f1)

# 4. Run
if __name__ == "__main__":

    model, metrics = train_and_evaluate(
        dataset_size=10000,
        vector_dim=16,
        hidden_dim=32,
        batch_size=32,
        epochs=50,
        lr=1e-3
    )
