import collections

def count_corpus(tokens):
    if len(tokens)==0 or isinstance(tokens[0],list):
        tokens=[token for line in tokens for token in line]
    
    return collections.Counter(tokens)

class Vocab:
    def __init__(self,tokens=None,min_freq=0,reserved_tokens=None):
        if tokens is None:
            tokens=[]
        
        if reserved_tokens is None:
            reserved_tokens=[]

        counter=count_corpus(tokens)

        self._token_freqs=sorted(counter.items(),key=lambda x:x[1],reverse=True)

        self.idx_to_token=['<unk>']+reserved_tokens
        self.token_to_idx={token:idx for idx,token in enumerate(self.idx_to_token)}

        for token,freq in self._token_freqs:
            if freq >= min_freq and token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token]=len(self.idx_to_token)-1
        
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self,tokens):
        if not isinstance(tokens,(list,tuple)):
            return self.token_to_idx.get(tokens,self.unk)
        return[self.__getitem__(token) for token in tokens]
    
    def totokens(self,indices):
        if not isinstance(indices,(list,tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[idx] for idx in indices]
    
    def unk(self):
        return 0
 
if __name__=='__main__':
    sentences=[
        ['hello', 'world', 'hello'],
        ['python', 'is', 'great'],
        ['hello', 'python']
    ]

    reversed_tokens=['<pad>', '<bos>', '<eos>']
    vocab=Vocab(tokens=sentences,min_freq=1,reserved_tokens=reversed_tokens)

    print("词表大小:", len(vocab))
    print("词表内容:", vocab.idx_to_token)
    print("词频排序:", vocab._token_freqs)

    print("\n'hello'的索引:", vocab['hello'])  
    print("['python', 'unknown']的索引:", vocab[['python', 'unknown']])  # 列表
    
    print("\n索引3对应的词:", vocab.totokens(3))
    print("索引列表[4, 5, 0]对应的词:", vocab.totokens([4, 5, 0]))
    
    print("\n未知词'unknown'的索引:", vocab['unknown'])
    print("未知词索引:", vocab.unk)