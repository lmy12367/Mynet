import torch
import torch.nn as nn
import torch.nn.functional as F

def get_params(vocab_size, num_hiddens, device):
    
    num_inputs = num_outputs = vocab_size
    
    def normal(shape):
        return torch.randn(shape, device=device) * 0.01
    
    
    W_xh = nn.Parameter(normal((num_inputs, num_hiddens)))
    W_hh = nn.Parameter(normal((num_hiddens, num_hiddens)))
    b_h = nn.Parameter(torch.zeros(num_hiddens, device=device))
    
   
    W_hq = nn.Parameter(normal((num_hiddens, num_outputs)))
    b_q = nn.Parameter(torch.zeros(num_outputs, device=device))
    
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch(nn.Module):
    
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = nn.ParameterList(get_params(vocab_size, num_hiddens, device))
        self.init_state = init_state
        self.forward_fn = forward_fn
    
    def forward(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)