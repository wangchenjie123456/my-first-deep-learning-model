import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#检查是否可以使用GPU加速
device=torch.device("cuda")if torch.cuda.is_available() else torch.device("cpu")

#导入数据
training_data=datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()

)

test_data=datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#创建数据加载器
train_dataloader=DataLoader(training_data,batch_size=64)
test_dataloader=DataLoader(test_data,batch_size=64)

#建立神经网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=nn.Flatten()
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    def forward(self,x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)
        return logits

#实例化    
model=NeuralNetwork().to(device)

#超参数
learning_rate=1e-3 #在每个批次/时期更新模型参数的量。较小的值会产生较慢的学习速度，而较大的值可能会导致训练期间的不可预测行为。
batch_size=64  #在参数更新之前通过网络传播的数据样本数
epochs=5  # 迭代数据集的次数


#训练函数和测试函数
def train_loop(dataloader,model,loss_fn,optimizer):
    size=len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    for batch,(X,y) in enumerate(dataloader):#遍历数据加载器：每次取一个 batch 的数据
        #迁移数据到CUDA
        X,y=X.to(device),y.to(device)

        # Compute prediction and loss
        #前向传播：把输入 X 送进模型，得到预测值 pred。
        #计算损失：用损失函数比较预测结果 pred 和真实标签 y，得到一个标量 loss。
        pred=model(X)
        loss=loss_fn(pred,y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #每100个打印一次日志
        if batch%100==0:
            loss, current=loss.item(),batch*len(X)
            print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader,model,loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss,correct=0,0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():#关闭反向传播，测试时不需要，以节省显存
        for X,y in dataloader:
            #迁移数据到CUDA
            X,y=X.to(device),y.to(device)

            pred=model(X)
            test_loss+=loss_fn(pred,y).item()
            #把 True/False 转成 1/0，求和，再转成 Python 数字，累计正确预测的样本数
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item()#

    test_loss/=num_batches#平均损失
    correct/=size#平均准确率
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#主程序
loss_fn=nn.CrossEntropyLoss()#损失函数：交叉熵损失
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)#优化器：随机梯度下降SGD

epochs=50#运行轮数

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

#保存模型
torch.save(model.state_dict(),"model.pth")

#加载模型
#model=NeuralNetwork()
#model.load_state_dict(torch.load("model.pth"))
#model.eval()
#model.to(device)
#model.eval()
