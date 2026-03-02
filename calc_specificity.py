import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from model import get_resnet_model
from data import get_dataloaders

def calc_specificity(y_true, y_pred, class_names, test_name):
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
    specificities = []
    
    print(f"\n{'='*80}")
    print(f"{test_name} - Specificity 计算")
    print(f"{'='*80}")
    print(f"{'类别':<10} | {'TP':>3} {'FP':>4} {'FN':>4} {'TN':>5} | {'Specificity':>12}")
    print("-" * 80)
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
        
        print(f"{class_names[i]:<10} | {tp:>3} {fp:>4} {fn:>4} {tn:>5} | {specificity:>12.4f}")
    
    print("-" * 80)
    print(f"{'Mean Specificity':<10} | {'':<18} | {np.mean(specificities):>12.4f}")
    print(f"{'Weighted Specificity':<10} | {'':<18} | {np.average(specificities, weights=cm.sum(axis=1)):>12.4f}")
    
    return specificities

# SetA
MODEL_PATH = 'resnet18_setA.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 修改 get_dataloaders 以支持 num_workers=0
import torch.utils.data as data_utils

def get_dataloaders_no_workers(train_dir, test_dir, batch_size=32):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
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

model = get_resnet_model(num_classes=5)
print(f"Loading model from {MODEL_PATH}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

# SetA Test
print("\n" + "="*80)
print("SetA 测试集")
print("="*80)
_, test_loader, class_names = get_dataloaders_no_workers(
    train_dir='./data/SetA/251_Train_A',
    test_dir='./data/SetA/251_Test_A',
    batch_size=32
)

all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

specificity_a = calc_specificity(all_labels, all_preds, class_names, "SetA (Test_A)")

# SetB Test
print("\n" + "="*80)
print("SetB 测试集")
print("="*80)
_, test_loader_b, _ = get_dataloaders_no_workers(
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

specificity_b = calc_specificity(all_labels_b, all_preds_b, class_names, "SetB (Test_B)")

# Additional Test
print("\n" + "="*80)
print("Additional 测试集")
print("="*80)
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

specificity_add = calc_specificity(all_labels_add, all_preds_add, class_names, "Additional (Test_NTU)")

print("\n" + "="*80)
print("汇总表 - Specificity")
print("="*80)
print(f"{'类别':<10} | {'SetA':>10} | {'SetB':>10} | {'Additional':>10}")
print("-" * 80)
for i, name in enumerate(class_names):
    print(f"{name:<10} | {specificity_a[i]:>10.4f} | {specificity_b[i]:>10.4f} | {specificity_add[i]:>10.4f}")
print("-" * 80)
print(f"{'Mean':<10} | {np.mean(specificity_a):>10.4f} | {np.mean(specificity_b):>10.4f} | {np.mean(specificity_add):>10.4f}")
