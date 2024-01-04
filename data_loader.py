from torchvision import datasets, transforms
import torch
import os
import torch.utils.data


def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose([
        transforms.Resize(256),  # 缩放图像
        transforms.RandomCrop(224),  # 随机裁剪到 224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose([
        transforms.Resize(256),  # 缩放图像
        transforms.CenterCrop(224),  # 中心裁剪到 224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, **kwargs)
    return test_loader
