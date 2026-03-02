import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model import get_resnet_model
from sklearn.metrics import classification_report
from tqdm import tqdm

# --- 配置参数 ---
TRAIN_DIR = './data/SetA/251_Train_A' # 你的训练集路径
TEST_DIR = './data/SetA/251_Test_A'   # 你的测试集路径
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1.以此获取数据
    train_loader, test_loader, class_names = get_dataloaders(TRAIN_DIR, TEST_DIR, BATCH_SIZE)
    print(f"Classes: {class_names}")
    
    # 2. 初始化模型
    model = get_resnet_model(num_classes=len(class_names)).to(DEVICE)
    
    # 3. 定义损失函数 (关键：解决不平衡)
    # 假设 class_names 排序是 ['G3', 'G4', 'G5', 'Normal', 'Stroma']
    # G5 只有 500 张，其他 800 张 -> 权重设大一点
    # 建议先运行一次 data.py 确认 G5 是第几个，这里假设是第 2 个索引
    weights = torch.tensor([1.0, 1.0, 1.6, 1.0, 1.0]).to(DEVICE) 
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # 4. 优化器
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 5. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # 添加进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # 实时更新进度条显示当前loss
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
        
        # (可选) 这里可以加一个简单的验证代码，每个epoch看一次准确率

    # 6. 最终评估 (生成报告用)
    print("Training Finished! Evaluating...")
    evaluate(model, test_loader, class_names)
    
    # 7. 保存模型
    save_path = "resnet18_setA.pth"
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")

def evaluate(model, loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 打印详细指标：F1, Precision, Recall
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == '__main__':
    main()
