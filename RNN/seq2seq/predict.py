import math
import collections
import torch
from data_preprocessing import Vocab
from net import EncoderDecoder, Seq2SeqEncoder, Seq2SeqDecoder

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    
    net.eval()
    
    tokens = src_sentence.lower().split(' ')
    tokens = [token for token in tokens if token]  #
    
  
    src_tokens = [src_vocab.token_to_idx.get(token, src_vocab.unk) for token in tokens]
    src_tokens = src_tokens + [src_vocab.token_to_idx['<eos>']]
    
   
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab.token_to_idx['<pad>'])
    
 
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    
  
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    
   
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    
   
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab.token_to_idx['<bos>']], dtype=torch.long, device=device), dim=0)
    
    output_seq = []
    for _ in range(num_steps):
        
        Y, dec_state = net.decoder(dec_X, dec_state)
        
       
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        
       
        if pred == tgt_vocab.token_to_idx['<eos>']:
            break
            
        output_seq.append(pred)
    
    
    return ' '.join(tgt_vocab.idx_to_token[i] for i in output_seq)

def bleu(pred_seq, label_seq, k):
    
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def truncate_pad(line, num_steps, padding_token):
    
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_steps = 10
    model_path = "D:/code/dp/dpV2/Seq2Seq/seq2seq_model.pth"
    
    
    checkpoint = torch.load(model_path, map_location=device)
    
    
    src_vocab = Vocab()
    src_vocab.idx_to_token = checkpoint['src_vocab'].idx_to_token
    src_vocab.token_to_idx = checkpoint['src_vocab'].token_to_idx
    
    tgt_vocab = Vocab()
    tgt_vocab.idx_to_token = checkpoint['tgt_vocab'].idx_to_token
    tgt_vocab.token_to_idx = checkpoint['tgt_vocab'].token_to_idx
    
    
    encoder = Seq2SeqEncoder(*checkpoint['encoder_params'])
    decoder = Seq2SeqDecoder(*checkpoint['decoder_params'])
    net = EncoderDecoder(encoder, decoder)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    
    
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    
    for eng, fra in zip(engs, fras):
        translation = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')