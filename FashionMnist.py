import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F

training_data = datasets.FashionMNIST(root = 'data', train = True, download = True, transform = Compose(
    [ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
test_data = datasets.FashionMNIST(root = 'data', train = False, download = True, transform = Compose(
    [ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    
print('the device is:', device)

class NueralNetwork(nn.Module):
    def __init__(self):
        super(NueralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax()
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack.forward(x)
        return logits

class MY_CNN(nn.Module):
    def __init__(self):
        super(MY_CNN, self).__init__()
        self.cnn_linear_stack = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 50, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.LogSoftmax(),
        )
    def forward(self, x):
        output = self.cnn_linear_stack(x)
        return output#记得要return,否则forward默认返回None,训练就会报'NoneType' object has no attribute 'log_softmax'

model = MY_CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

def create_acc_dict(class_num: int):
    acc_dict = {}
    init_acc_list = [0, 0] #前一个数表示预测正确的数量，后一个数表示这个标签在这个训练集中一共有多少个
    for i in range(class_num):
        acc_dict[i] = copy.deepcopy(init_acc_list) #初始acc_dict = {0 : [0, 0], 1 : [0, 0], ..., 9 : [0, 0]}
    return acc_dict

def compute_acc(pred, label, acc_dict):
    
    correct_or_not = pred.argmax(1) == label
    
    for i in range(len(correct_or_not)):
        if correct_or_not[i] == True:
            number = label[i].item() #这里的label[i]是一个tensor，所以要用.item()把它变成一个数
            acc_dict[number][0] += 1
            acc_dict[number][1] += 1
        else:
            acc_dict[label[i].item()][1] += 1  #这里的.item()也一样

def statistical_acc(acc_dict):
    statistical_acc = {}
    avg = 0
    for i in range(len(acc_dict)):
        statistical_acc[i] = str(acc_dict[i][0] / acc_dict[i][1]) + '%'
        avg += acc_dict[i][0] / acc_dict[i][1]
    statistical_acc['avg'] = str(avg / 10) + '%'
    return statistical_acc            
            
def train(dataloader, model, loss_fn, optimizer):
    train_acc_dict = create_acc_dict(class_num = 10)
    size = len(dataloader.dataset)
    for batch, (x,y) in enumerate(dataloader):
        
        x, y = x.to(device), y.to(device)
        
        pred = model(x)
        loss = loss_fn(pred, y)
        compute_acc(pred, y, train_acc_dict)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
    stat_train_acc_dict = statistical_acc(train_acc_dict)                    
    print(train_acc_dict,'\n',stat_train_acc_dict)

def test(dataloader, model):
    test_acc_dict = create_acc_dict(class_num = 10)
    size = len(dataloader.dataset)
    model.eval()  #表示接下来进入ｅｖａｌｕａｔｉｏｎ模式
    test_loss, correct = 0, 0
    with torch.no_grad():#torch.no_grad()这个函数表示接下来的计算不用构建计算图（pytorch中tensor的计算默认会构建计算图）
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            compute_acc(pred, y, test_acc_dict)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()#(pred.argmax(1) == y)比较是否相等，返回的
#是tensor（具体来说是[True/False]的类型，大小为[batchsize * 1]）,接下来用了.type 将True/False转换成一个数           
    test_loss /= size
    correct /= size
    stat_test_acc_dict = statistical_acc(acc_dict = test_acc_dict)    
    print(f'test error: \n accuracy: {(100 * correct):0.1f}%, avg loss: {test_loss:>8f} \n')

epochs = 20
for t in range(epochs):
    print(f'epoch{t + 1}\n--------------------------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print('Done!')
