import torch
from net import RNNModelScratch, get_params, init_gru_state,gru
from data_preprocessing import Vocab
import os

def predict_ch8(prefix, num_preds, net, vocab, device):

    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).item()))
    
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def load_model(model_path, device):

    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']
    
    net = RNNModelScratch(
        checkpoint['vocab_size'], 
        checkpoint['num_hiddens'], 
        device,
        get_params,
       init_gru_state,
       gru
    )
    
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    net.eval()
    
    return net, vocab

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_dir = "D:\\code\\dp\\dpV2\\GRU"
    model_path = os.path.join(model_dir, "GRU_model.pth")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请先运行 train.py 训练模型")
        return
    
    print(f"加载模型: {model_path}")
    model, vocab = load_model(model_path, device)
    
    print("\n预测示例:")
    print(predict_ch8('time traveller', 10, model, vocab, device))
    print(predict_ch8('traveller', 10, model, vocab, device))
    
    while True:
        prefix = input("\n输入起始文本 (输入 'exit' 退出): ")
        if prefix.lower() == 'exit':
            break
        try:
            num_preds = int(input("输入要预测的字符数: "))
            prediction = predict_ch8(prefix, num_preds, model, vocab, device)
            print(f"完整文本: {prediction}")
            print(f"预测部分: {prediction[len(prefix):]}")
        except Exception as e:
            print(f"预测出错: {e}")

if __name__ == '__main__':
    main()