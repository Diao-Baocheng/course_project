# predict.py
import torch
from model import get_resnet_model        # 导入你的模型结构
from data import get_dataloaders          # 导入数据处理
from sklearn.metrics import classification_report

# 配置
MODEL_PATH = 'resnet18_setA.pth'          # 刚才保存的文件
TEST_DIR = './data/SetA/SetB/251_Test_B'  # 你想测试的任何数据集 (Set B)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict():
    # 1. 准备“空壳”模型 (结构必须和训练时一模一样)
    # 注意：这里 num_classes 必须和你训练时一样 (5类)
    model = get_resnet_model(num_classes=5) 
    
    # 2. 加载“记忆” (权重)
    print(f"正在加载模型: {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    
    # 3. 开启“考试模式” (Eval Mode)
    # 这步极其重要！它会固定住 BatchNorm 和 Dropout，否则预测结果会乱跳
    model.eval()

    # 4. 准备数据 (不需要 Train loader 了)
    # 注意：这里我们利用之前的函数，但只取 test_loader
    # 你可能需要微调 data.py 让它支持只返回 test_loader，或者简单的 dummy 传参
    _, test_loader, class_names = get_dataloaders(
        train_dir='./data/SetA/251_Train_A', # 这里路径随便填，反正不用
        test_dir=TEST_DIR, 
        batch_size=32
    )
    
    # 5. 开始预测
    all_preds = []
    all_labels = []
    
    print("开始预测...")
    with torch.no_grad(): # 考试时不需要求导，节省显存
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 6. 输出报告
    print(f"在数据集 {TEST_DIR} 上的表现:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == '__main__':
    predict()