# Mynet
《动手学深度学习》网络复现
---

## 项目概述
本仓库复现了李沐老师《动手学深度学习》中的经典CNN网络，所有实现均使用PyTorch框架。目前已实现以下网络架构：
- **LeNet**
- **AlexNet**
- **VGG**
- **NiN** 
- **GoogLeNet/Inception**
- **SSD**

每个网络均包含独立实现：
- 📁 `net.py`：网络模型定义
- ⚙️ `train.py`：训练脚本
- 🧪 `predict.py`：推理脚本(暂未实现)

## 关键配置说明

### 数据集路径配置 (必须修改)

```python
train_dataset = datasets.MNIST(
    root='../data/mnist',  # ← 修改此处
    train=True,
    download=False,
    transform=transform
)
```

```python
test_dataset = datasets.MNIST(
    root='../data/mnist',  # ← 修改此处
    train=False,
    download=False,
    transform=transform
)
```

### 批量大小调整 (推荐修改)

根据GPU显存调整 (默认值较小)

```python
batch_size = 64  # ← 可增大至128/256等
```

### 修改模型保存位置 (默认保存到当前目录)
```python
def save_model(model, filename="../model.pth"):  # ← 修改此处
    try:
        torch.save(model.state_dict(), filename)
        print(f"模型已保存至 {filename}")
    except Exception as e:
        print(f"保存失败: {e}")
```

### 样本抽样设置 (快速验证)

```python
# 默认只使用512个训练样本和128个测试样本
train_dataset = torch.utils.data.Subset(train_dataset, range(512))  # ← 删除此行使用完整数据集
test_dataset = torch.utils.data.Subset(test_dataset, range(128))    # ← 删除此行使用完整数据集
```

