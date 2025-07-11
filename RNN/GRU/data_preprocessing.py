import collections
import re
import os
import requests
import torch


def download_time_machine():
    url = 'https://raw.githubusercontent.com/d2l-ai/d2l-en/master/data/timemachine.txt'
    save_path = 'D:\\code\\document\\data\\timemachine.txt'
    
    if not os.path.exists(save_path):
        print("Downloading dataset...")
        response = requests.get(url)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    return save_path

def read_time_machine():
    
    file_path = download_time_machine()
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        raise ValueError('未知词元类型: ' + token)

class Vocab:
    
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
            
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

    @staticmethod
    def count_corpus(tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1, token='char'):
    lines = read_time_machine()
    tokens = tokenize(lines, token)
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    corpus = corpus[torch.randint(0, num_steps - 1, (1,)).item():]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    torch.manual_seed(42)
    initial_indices = torch.tensor(initial_indices)[torch.randperm(len(initial_indices))].tolist()
    
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i:i+batch_size]
        X = [corpus[j:j+num_steps] for j in initial_indices_per_batch]
        Y = [corpus[j+1:j+1+num_steps] for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = torch.randint(0, num_steps, (1,)).item()
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset+num_tokens])
    Ys = torch.tensor(corpus[offset+1:offset+1+num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i+num_steps]
        Y = Ys[:, i:i+num_steps]
        yield X, Y

class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter=False, max_tokens=10000, token='char'):
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens, token)
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.use_random_iter = use_random_iter
        
    def __iter__(self):
        if self.use_random_iter:
            return seq_data_iter_random(self.corpus, self.batch_size, self.num_steps)
        return seq_data_iter_sequential(self.corpus, self.batch_size, self.num_steps)