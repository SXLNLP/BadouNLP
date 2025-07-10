import time
import torch
from config import config


# 训练模型的函数
def train_model(model, train_loader, val_loader, vocab_size, device):
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 记录训练历史
    train_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0
    best_model = None

    print(f"开始训练{model.__class__.__name__}模型...")
    for epoch in range(config.num_epochs):
        start_time = time.time()

        # 训练模式
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        # 遍历训练数据
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # 记录训练损失和准确率
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # 计算平均损失和准确率
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # 验证模式
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        # 不计算梯度，节省内存和时间
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # 记录本轮结果
        train_history['train_loss'].append(train_loss)
        train_history['val_loss'].append(val_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        # 打印训练进度
        print(f'第 {epoch + 1}/{config.num_epochs} 轮')
        print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc * 100:.2f}%')
        print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc * 100:.2f}%')
        print(f'  本轮耗时: {epoch_mins:.0f}分{epoch_secs:.0f}秒')

    return train_history, best_model


# 评估模型预测速度
def evaluate_speed(model, val_loader, device):
    model = model.to(device)
    model.eval()

    # 取一个小批量数据测试速度
    with torch.no_grad():
        batch = next(iter(val_loader))
        input_ids = batch['input_ids'].to(device)

        for _ in range(5):
            _ = model(input_ids)

        # 计时
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        for _ in range(100):
            _ = model(input_ids)
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()

    # 计算每秒处理的样本数
    infer_time = (end_time - start_time) / 100
    infer_speed = config.batch_size / infer_time  # 样本/秒
    return infer_speed
