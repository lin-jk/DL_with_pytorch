import torch
import torchvision
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import ToTensor, Lambda, Compose
import struct
import os
import numpy as np
import copy
from torch import nn

lass mnist_dataset(Dataset):
    def __init__(self, mode: str, transform):#两种初始化方法，把数据全读进来，或者在__getitem__里面去打开文件然后读数据,这里用了第一种方法
        self.filepath = '/home/ljk/Downloads/binary_data'
        self.mode = mode
        self.transform = transform
        
        if self.mode == 'train':
            with open(self.filepath + '/train-images.idx3-ubyte', mode = 'rb') as train_img_file:
                magic, num_of_train_imgs, rows, columns = struct.unpack('>IIII', train_img_file.read(16))
                self.train_data = np.fromfile(train_img_file, dtype = np.uint8).reshape(num_of_train_imgs, 28, 28)
            with open(self.filepath + '/train-labels.idx1-ubyte', mode = 'rb') as train_label_file:
                magic, num_of_train_label = struct.unpack('>II', train_label_file.read(8))
                self.train_label = np.fromfile(train_label_file, dtype = np.uint8)
                
        elif self.mode == 'test':
            with open(self.filepath + '/t10k-images.idx3-ubyte', mode = 'rb') as test_img_file:
                magic, num_of_test_imgs, rows, columns = struct.unpack('>IIII', test_img_file.read(16))
                self.test_data = np.fromfile(test_img_file, dtype = np.uint8).reshape(num_of_test_imgs, 28, 28)#注意格式，不是２８＊２８了
            with open(self.filepath + '/t10k-labels.idx1-ubyte', mode = 'rb') as test_label_file:
                magic, num_of_test_label = struct.unpack('>II', test_label_file.read(8))
                self.test_label = np.fromfile(test_label_file, dtype = np.uint8)    
        
    def __getitem__(self, index):
        if self.mode == 'train' :
            data, label = self.train_data[index], self.train_label[index]
        elif self.mode == 'test':
            data, label = self.test_data[index], self.test_label[index]
        
        if self.transform is not None:#ToTensor会对np.ndarray做标准化，然后转换成tensor!!!!重点是标准化
            data = self.transform(data)
        
        return data, label
    
    def __len__(self):
        if self.mode == 'train':
            return self.train_data.shape[0]
        elif self.mode == 'test':
            return self.test_data.shape[0]
          
train_data = mnist_dataset('train', ToTensor())#注意这里传进去的是ToTensor(),而不是ToTensor,就是说传进去的是类的实例对象
test_data = mnist_dataset('test', ToTensor())

batch_size = 64
train_dataloader = DataLoader(train_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

# for x, y in test_dataloader:
#     print(x.shape, y.shape)
#     break
    
class MY_MNIST_NET(torch.nn.Module):
    def __init__(self):
        super(MY_MNIST_NET, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        output = self.linear(x)
        return output

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
    
device = 'cuda'
model = MY_CNN().to(device)
# print(model)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)

def create_acc_dict(class_num: int):
    acc_dict = {}
    init_acc_list = [0, 0] #前一个数表示预测正确的数量，后一个数表示这个标签在这个训练集中一共有多少个
    for i in range(class_num):
        acc_dict[i] = copy.deepcopy(init_acc_list) #初始acc_dict = {0 : [0, 0], 1 : [0, 0], ..., 9 : [0, 0]}
    return acc_dict                     #注意这里用了copy.deepcopy函数,否则的话acc_dict中的列表其实都是init_acc_list
        #如果用acc_dict[i] = init_acc_list，即直接用init_acc_list来赋值，会导致所有列表都指向同一个内存空间
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
    size = len(dataloader.dataset)
    train_acc_dict = create_acc_dict(class_num = 10)
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        compute_acc(output, y, train_acc_dict)
        loss = loss_fn(output, y.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print('loss:', loss, batch * len(x), '/', size)
    stat_train_acc_dict = statistical_acc(train_acc_dict)        
    print(train_acc_dict,'\n',stat_train_acc_dict)
            
def test(dataloader, model, loss_fn):
    test_acc_dict = create_acc_dict(class_num = 10)
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            compute_acc(output, y, test_acc_dict)
    stat_test_acc_dict = statistical_acc(acc_dict = test_acc_dict)
    print('-------------------------------test---------------------------------------------------')        
    print(test_acc_dict, '\n', stat_test_acc_dict)
    
epoch = 20
for i in range(epoch):
    print(f'Epoch{i + 1}\n--------------------------------------------')
    train(dataloader = train_dataloader, model = model, loss_fn = loss_fn, optimizer = optimizer)
    test(dataloader = test_dataloader, model = model, loss_fn = loss_fn)
    
print('Done!')
