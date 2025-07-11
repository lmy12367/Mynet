import torch
import math
import time
from net import RNNModelScratch, get_params, init_rnn_state, rnn
from data_preprocessing import SeqDataLoader
import matplotlib.pyplot as plt
import os
import torch.nn as nn

def grad_clipping(net, theta):
    """梯度裁剪"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    """训练一个epoch"""
    state, timer = None, time.time()
    metric = [0.0, 0]  # 训练损失之和, 词元数量
    
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 对于自定义模型，断开状态与计算图的连接
            state = [s.detach() for s in state] if isinstance(state, list) else state.detach()
        
        X, Y = X.to(device), Y.to(device)
        y_hat, state = net(X, state)
        y = Y.T.reshape(-1)
        l = loss(y_hat, y.long()).mean()
        
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        
        metric[0] += l.item() * y.numel()
        metric[1] += y.numel()
    
    return math.exp(metric[0] / metric[1]), metric[1] / (time.time() - timer)

def train_rnn():
    """训练RNN模型"""
    # 配置参数
    batch_size, num_steps = 32, 35
    num_hiddens = 256
    lr = 1
    num_epochs = 500
    use_random_iter = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"使用设备: {device}")
    
    # 加载数据
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens=10000)
    vocab = data_iter.vocab
    
    # 初始化模型
    net = RNNModelScratch(
        len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)
    
    # 转移模型到设备
    net.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # 损失函数和优化器
    loss = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr)
    
    # 训练循环
    perplexities = []
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(
            net, data_iter, loss, updater, device, use_random_iter)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Perplexity: {ppl:.1f}, Speed: {speed:.1f} tokens/sec')
            perplexities.append(ppl)
    
    # 保存模型
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "D:\\code\\dp\\Review-DP\\dpV2\\RNN\\rnn_model.pth")
    
    torch.save({
        'model_state_dict': net.state_dict(),
        'vocab_size': len(vocab),
        'num_hiddens': num_hiddens,
        'vocab': vocab
    }, model_path)
    
    print(f"模型已保存至: {model_path}")
    
    # 绘制困惑度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(10, num_epochs+1, 10), perplexities, 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training Perplexity')
    plt.grid(True)
    plt.savefig('perplexity.png')
    plt.show()

if __name__ == '__main__':
    train_rnn()