import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import confusion_matrix
from data_loader import load_training, load_testing
from DALN import DALN
import numpy as np
import random


# Set random seed for reproducibility
def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed_value)
    random.seed(seed_value)

set_seed(42)

# Rest of your parameters and settings
num_classes = 8

# 设定超参数
num_classes = 8
batch_size = 32
learning_rate = 0.003
num_epochs = 50
momentum = 0.9
weight_decay = 0.0005
trade_off_lambda = 0.5  # 主要调这个参数，非常重要
lr_decay_step = 3
lr_decay_gamma = 0.5
root_path = 'dataset'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
source_loader = load_training(root_path, 'data20230531', batch_size, kwargs)
target_loader = load_training(root_path, 'data20230606', batch_size, kwargs)
target_test_loader = load_testing(root_path, 'data20230606', batch_size, kwargs)

# 初始化模型
model = DALN(num_classes=num_classes).to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

# 用于保存最佳模型的设置
best_accuracy = 0.0
best_model_path = 'best_model0531_0606.pth'

# 训练循环
for epoch in range(num_epochs):
    model.train()
    len_dataloader = min(len(source_loader), len(target_loader))
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    total_loss = 0.0

    for i in range(len_dataloader):
        source_data, source_label = next(source_iter)
        target_data, _ = next(target_iter)
        source_data, source_label = source_data.to(device), source_label.to(device)
        target_data = target_data.to(device)

        # 前向传播和损失计算
        loss = model(source_data, source_label, target_data, trade_off_lambda)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len_dataloader
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    scheduler.step()

    # 测试循环
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in target_test_loader:
            images, labels = images.to(device), labels.to(device)
            _, outputs = model.classifier(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * np.mean(np.array(all_labels) == np.array(all_predictions))
    print(f'Epoch [{epoch+1}/{num_epochs}]: Accuracy on target test set: {accuracy:.2f}%')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f'New best model saved with accuracy: {accuracy:.2f}%')

        # 当新的最佳模型出现时，计算并打印混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        print(f'Confusion Matrix for the best model at Epoch [{epoch+1}/{num_epochs}]:\n{cm}')

print('Training completed.')
