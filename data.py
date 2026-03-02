import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(train_dir, test_dir, batch_size=32):
    # 1. 定义训练集的增强策略 (Augmentation)
    # 目的：让模型看过"各种姿势"的癌症切片，防止死记硬背
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),      # 必须：压缩尺寸
        transforms.RandomHorizontalFlip(),  # 增强：水平翻转
        transforms.RandomVerticalFlip(),    # 增强：垂直翻转
        transforms.RandomRotation(15),      # 增强：随机旋转15度
        transforms.ToTensor(),              # 必须：转为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225]) # ImageNet标准化
    ])

    # 2. 定义测试集的预处理
    # 注意：测试集只要"看清楚"就行，不要乱动
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ])

    # 3. 加载数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # 4. 打印一下类别索引，确保 G5 的位置 (用来后面设权重)
    # print(f"Class mapping: {train_dataset.class_to_idx}")

    # 5. 打包成 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader, train_dataset.classes