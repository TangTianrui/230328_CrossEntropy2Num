import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

#注释过的代码为GPU加速前版本
batch_size=50#学习和测试的集合大小
learning_rate=0.01#学习率
epochs=10#代数

##数据导入
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)

w1,b1=torch.randn(200,784,requires_grad=True),torch.zeros(200,requires_grad=True)
w2,b2=torch.randn(200,200,requires_grad=True),torch.zeros(200,requires_grad=True)
w3,b3=torch.randn(100,200,requires_grad=True),torch.zeros(100,requires_grad=True)
w4,b4=torch.randn(10,100,requires_grad=True),torch.zeros(10,requires_grad=True)




#初始化w参数
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)
torch.nn.init.kaiming_normal_(w4)

def forward(x):
    #构造三层简单前向网络
    x=x@w1.t()+b1
    x=torch.relu(x)
    x=x@w2.t()+b2
    x=torch.relu(x)
    x=x@w3.t()+b3
    x=torch.relu(x)
    x=x@w4.t()+b4
    x=torch.relu(x)
    return x
#
# class MLP(nn.Module):
#     def __init__(self):
#         #复写自身及继承
#         super(MLP,self).__init__()
#         #构造网络结构和激活函数
#         self.model=nn.Sequential(
#             nn.Linear(784,200),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(200,200),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(200,10),
#             nn.LeakyReLU(inplace=True)
#         )
#
#     def forward(self,x):
#         x=self.model(x)
#         return x
# device=torch.device('cuda:0')
# #选择设备运行环境，即cuda0，显卡的1核
# net=MLP().to(device)
#创建网络并移动到指定运行环境中，即显卡中运行
optimizer=optim.SGD([w1,b1,w2,b2,w3,b3,w4,b4],learning_rate)
# optimizer=optim.SGD(net.parameters(),lr=learning_rate)
loss_function=nn.CrossEntropyLoss()
# loss_function=nn.CrossEntropyLoss().to(device)
#将损失函数移动由GPU运行

for epoch in range(epochs):
    for batch, (data,target) in enumerate(train_loader,1):
        #从train_loader(训练集)中一个个取出各序列号和其对应的键值对：
        data=data.view(-1,28*28)
        # data,target=data.to(device),target.to(device)
        # logits=net(data)
        logits=forward(data)#传统前向方法，没有指定由GPU运行
        func=loss_function(logits,target)
        #优化器迭代
        optimizer.zero_grad()
        func.backward()
        optimizer.step()
        # if batch%100==0:
        #     print('Train Epoch{}:{}/{},Loss={:.6f}'.format(epoch,batch*len(data),len(train_loader.dataset),func.item()))

    test_func=0
    correct=0
    for data,target in test_loader:
        #通过测试集检测模型的准确性
        data=data.view(-1,28*28)
        # data,target=data.to(device),target.to(device)
        # logits=net(data)
        logits=forward(data)
        test_func+=loss_function(logits,target).item()
        #通过累加求测试集的总cost
        pred=logits.data.max(1)[1]#最大概率确定预测值
        correct+=pred.eq(target.data).sum()#求和总正确数量

    test_func/=len(test_loader.dataset)#求平均损失函数loss
    print('Test{} has the average loss = {:.4f},correct rate = {}/{}'.format(epoch,test_func,correct,len(test_loader.dataset)))















