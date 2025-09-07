import torch
import numpy as np
data=[[1,2],[3,4]]
x_data=torch.tensor(data)
np_array=np.array(data)
x_np=torch.from_numpy(np_array)
x_zeros=torch.zeros_like(x_data)
#print(x_zeros)

tensor =torch.ones(4,4,device='cuda')
#print(tensor[0])
#print(tensor[:,0])
#print(tensor[...,-1])
tensor[1]=0
#print(tensor)

t1=torch.cat([tensor,tensor,tensor],dim=1)
print(t1)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
#以上内容：@是矩阵乘法，matmul是矩阵乘法，matmul可以指定输出，out=y3指输出到y3中

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
#以上，*是逐个元素乘法，z3[i][j]=z1[i][j]*z2[i][j]

