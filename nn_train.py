import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

if(torch.accelerator.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#看cuda加速是否可行，如果可行则使用cuda加速
#print(device)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        #nn.Flatten() = “把图片展平” 的一层，常常用在卷积层（CNN）和全连接层（Linear）之间。
        self.flatten=nn.Flatten()

        #相当于一个三层全连接的神经网络（中间带 ReLU 激活）：输入 (28*28) → 全连接层(512) → ReLU → 全连接层(512) → ReLU → 全连接层(10)
        #ReLU是max(0,x)
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )#sequential会把输入依次通过每一层

    def forward(self,x): 
         # 定义前向传播方法，x为输入数据
         x=self.flatten(x) # 1. 把输入图片展平成向量
         logits=self.linear_relu_stack(x)# 2. 送入全连接 + ReLU 堆叠，得到输出
         return logits# 3. 返回未经过 softmax 的结果
    
model=NeuralNetwork().to(device)#实例化
#print(model)

###Do not call model.forward() directly!!! Call model(x) instead.

#Softmax = 把模型输出的原始分数转成概率，让我们知道模型认为每个类别的可能性有多大。

x=torch.rand(1,28,28).to(device)#随机生成一张“假图片”。
logits=model(x)#送入模型，得到输出
pred_prob=nn.Softmax(dim=1)(logits)#把输出转成概率
y_pred=torch.argmax(pred_prob,dim=1)#取概率最大的类别作为预测结果
print(y_pred)

#输出模型结构
#print(f"Model structure: {model}\n\n")

#for name, param in model.named_parameters():
#    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")






