import collections

class Vocab:
    def __init__(self,tokens=None,min_freq=0,reserved_tokens=None):
        if tokens in None:
            tokens=[]
        
        if reserved_tokens is None:
            reserved_tokens=[]

        
        