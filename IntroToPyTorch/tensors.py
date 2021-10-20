# Tensors
import torch
import numpy as np

# https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html

# Tensors are similar to NumPy's ndarrays, except that
# it can run on GPUs or other hardware accelerators

#################################################################
# initializing Tensor
#   Directly from data
data = [[1, 2],[3, 4]] 
x_data = torch.tensor(data)

#   From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#   From another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")


x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

#   with randomm or constant values
shape = (2,3,) # shape is a tuple of tensor dimmensions
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#################################################################3
# Attributes of Tensor
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Operations on Tensors
# https://pytorch.org/docs/stable/torch.html
# these operators can be run on the GPU

# by default, tensors are created on the CPU, so we need to mmove tensors to GPU using .to method
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

#   standard numpy-lie indexing and slicing
tensor = torch.ones(4,4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[...,-1])
tensor[:,1] = 0
print(tensor)

#   Joining two tesnors (concatenation? )
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#   Arithmetic operations
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#   single-element tensors
# you ccan convert one-element tensor to a Python numerical value using item() by aggregating all values of a tensor into one value
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#   In-place operations
# Operations that store the result into the operand are called in-place. They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.
# operator assignment like += and *= ??
print(tensor, "\n")
tensor.add_(5)
print(tensor)

#################################################################
# Bridge with Numpy
# tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

# so NumPy and Tensor works simultaneously...