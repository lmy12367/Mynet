import os
import time
import torch
import matplotlib.pyplot as plt
from data_preprocessing import load_data_nmt
from net import EncoderDecoder, Seq2SeqEncoder, Seq2SeqDecoder, MaskedSoftmaxCELoss, grad_clipping

def train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device, save_path):

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    
  
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
  
    train_loss = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_tokens = 0
        start_time = time.time()
        
        for batch in train_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            
           
            bos = torch.tensor([tgt_vocab.token_to_idx['<bos>']] * Y.shape[0], 
                              device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            
         
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            
           
            optimizer.zero_grad()
            l.sum().backward()
            grad_clipping(net, 1)
            optimizer.step()
            
            
            with torch.no_grad():
                epoch_loss += l.sum().item()
                num_tokens += Y_valid_len.sum().item()
        
        
        avg_loss = epoch_loss / num_tokens
        train_loss.append(avg_loss)
        epoch_time = time.time() - start_time
        
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, '
                  f'Time: {epoch_time:.1f} sec, Tokens/sec: {num_tokens/epoch_time:.1f}')
    
  
    torch.save({
        'model_state_dict': net.state_dict(),
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'encoder_params': (len(src_vocab), embed_size, num_hiddens, num_layers, dropout),
        'decoder_params': (len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout),
        'train_loss': train_loss
    }, save_path)
    
    print(f'模型已保存至: {save_path}')
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs+1), train_loss, 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    loss_plot_path = os.path.join(os.path.dirname(save_path), 'training_loss.png')
    plt.savefig(loss_plot_path)
    plt.show()
    
    return train_loss

if __name__ == '__main__':
    
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_save_path = "D:/code/dp/dpV2/Seq2Seq/seq2seq_model.pth"
    
    print(f"使用设备: {device}")
    
    
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    print(f"源语言词表大小: {len(src_vocab)}, 目标语言词表大小: {len(tgt_vocab)}")
    
    
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    
   
    train_loss = train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device, model_save_path)