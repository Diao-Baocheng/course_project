import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from model import get_resnet_model  # 确保你导入了正确的模型定义
from data import get_dataloaders    # 确保导入了数据加载器

# --- 配置 ---
MODEL_PATH = 'resnet18_setA.pth'       # 你保存的模型路径
TEST_DIR = './data/SetA/251_Test_A'    # 核心测试集路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

def evaluate_model():
    # 1. 准备数据 (只取 test_loader)
    # 注意：这里我们调用 get_dataloaders 但只关心第二个返回值
    _, test_loader, class_names = get_dataloaders(
        train_dir='./data/SetA/251_Train_A', # 占位符，不会用到
        test_dir=TEST_DIR, 
        batch_size=BATCH_SIZE
    )
    
    # 2. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    model = get_resnet_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # 开启评估模式 (关闭 Dropout/BatchNorm 更新)

    # 3. 预测循环
    y_true = []
    y_pred = []
    
    print("Running inference...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # --- 4. 生成报告数据 (Report Deliverables) ---
    
    # A. 打印文本报告 (F1, Precision, Recall/Sensitivity)
    print("\n" + "="*50)
    print(f"Classification Report for {TEST_DIR}")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # B. 画混淆矩阵 (Confusion Matrix) 
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Set A')
    
    # 保存混淆矩阵图片
    cm_filename = 'confusion_matrix_setA.png'
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    print(f"\n混淆矩阵已保存到: {cm_filename}")
    plt.close()

if __name__ == '__main__':
    evaluate_model()