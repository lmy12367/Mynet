import re,collections,os,urllib.request

URL='https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
File='./dp/Mynet\RNN/Sequence_model/timemachine.txt'

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