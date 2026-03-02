import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from model import get_resnet_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MODEL_PATH = 'resnet18_setA.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存: {filename}")

def get_dataloaders_no_workers(train_dir, test_dir, batch_size=32):
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader, train_dataset.classes

# 加载模型
model = get_resnet_model(num_classes=5)
print(f"加载模型: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

# SetB
print("\n生成 SetB 混淆矩阵...")
_, test_loader_b, class_names = get_dataloaders_no_workers(
    train_dir='./data/SetA/251_Train_A',
    test_dir='./data/SetA/SetB/251_Test_B',
    batch_size=32
)

all_preds_b = []
all_labels_b = []
with torch.no_grad():
    for inputs, labels in test_loader_b:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds_b.extend(preds.cpu().numpy())
        all_labels_b.extend(labels.numpy())

plot_confusion_matrix(all_labels_b, all_preds_b, class_names, 
                     'Confusion Matrix - SetB (Test_B)', 'confusion_matrix_setB.png')

# Additional
print("\n生成 Additional 混淆矩阵...")
_, test_loader_add, _ = get_dataloaders_no_workers(
    train_dir='./data/SetA/251_Train_A',
    test_dir='./data/SetA/Test_NTU_additional',
    batch_size=32
)

all_preds_add = []
all_labels_add = []
with torch.no_grad():
    for inputs, labels in test_loader_add:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds_add.extend(preds.cpu().numpy())
        all_labels_add.extend(labels.numpy())

plot_confusion_matrix(all_labels_add, all_preds_add, class_names, 
                     'Confusion Matrix - Additional (Test_NTU)', 'confusion_matrix_additional.png')

print("\n所有混淆矩阵已生成完毕！")
