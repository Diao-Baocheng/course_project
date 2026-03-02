import matplotlib.pyplot as plt
import re

# 从 training.log 中提取 Epoch 和 Loss 数据
log_file = 'training.log'
epochs = []
losses = []

with open(log_file, 'r') as f:
    for line in f:
        # 查找格式: "Epoch 1/20, Average Loss: 0.xxxx"
        match = re.search(r'Epoch (\d+)/20, Average Loss: ([\d.]+)', line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            epochs.append(epoch)
            losses.append(loss)

if epochs and losses:
    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, losses, 'b-o', linewidth=2, markersize=6, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss vs Epoch (ResNet18 on SetA)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('training_loss_curve.png', dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存: training_loss_curve.png")
    print(f"\n提取的数据:")
    print(f"Epochs: {epochs}")
    print(f"Losses: {[f'{l:.4f}' for l in losses]}")
else:
    print("未找到有效的 Epoch 和 Loss 数据")
