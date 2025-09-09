import re,collections,os,urllib.request

from Vocab import Vocab

URL='https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
File='./dp/Mynet/RNN/Sequence_model/timemachine.txt'

if not os.path.exists(File):
    urllib.request.urlretrieve(URL,File)

def read_time_machine():
    with open(File,'r',encoding='utf-8') as f:
        lines=f.readlines()

    return [re.sub(r'[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines=read_time_machine()
print("total",(len(lines)))
print("first",lines[0])

def tokenize(lines,token='word'):
    if token=='word':
        return [line.split() for line in lines]
    
    elif token=='char':
        return [list(line) for line in lines]
    
    else:
        print('erro'+token)

tokens=tokenize(lines)
for i in range(11):
    print(tokens[i])

def count_corpus(tokens):
    if len(tokens)==0 or isinstance(tokens[0],list):
        tokens=[token for line in tokens for token in line]
    
    return collections.Counter(tokens)

test_tokens = [['hello', 'world'], ['hello', 'python']]
print(count_corpus(test_tokens))

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:20])

for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])

