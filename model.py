import torch.nn as nn
from torchvision import models

def get_resnet_model(num_classes=5):
    # 1. 加载预训练的 ResNet18
    # weights='DEFAULT' 等同于 pretrained=True
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 2. 修改最后的全连接层 (Fully Connected Layer)
    # 原始 ResNet 输出是 1000 类，我们要改成 5 类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model