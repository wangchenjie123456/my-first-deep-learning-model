import torch

###y=w*x+b,w是权重参数，b是偏置参数，w和b是学习的参数
x=torch.ones(5)#input tensor
y=torch.zeros(3)#expected output
w=torch.randn(5,3,requires_grad=True)#weights
b=torch.randn(3,requires_grad=True)#bias
#requires_grad=True表示需要计算梯度，因为要计算损失函数对w和b的梯度，从而更新w和b
#您可以在创建张量时设置 requires_grad 的值，也可以稍后使用 x.requires_grad_（True） 方法设置值。

z=torch.matmul(x,w)+b#forward pass
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
#binary_cross_entropy_with_logits 会先对 z 做 sigmoid，再和 y 计算二分类交叉熵。

#交叉熵（Cross Entropy）是用来衡量 两个概率分布之间差异 的。BCE：二分类交叉熵
#pytorch提供了nn.BCEWithLogitsLoss：更常用，直接输入 logits（原始输出），内部会先做 sigmoid 再算 BCE，数值更稳定
#二分类交叉熵就是用来衡量模型预测的概率和真实标签之间差距的损失函数。预测越准，交叉熵越小。

#z的梯度=z.grad_fn

loss.backward()#反向传播，计算w和b的梯度，以优化神经网络中的权重和偏重

print("w.grad=",w.grad)

print("b.grad=",b.grad)