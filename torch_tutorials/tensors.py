import torch
import numpy as np

# tensors can be made directly from data
# data type is automatically inferred
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# tensors from numpy arrays
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# these new tensors retains the (shape, datatype) of the argument tensor
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones-like Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random-like Tensor: \n {x_rand} \n")


# with random or constant values
shape = (
    2,
    3,
)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

###
# attributes of a tensor

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

###
# operations on tensors

# by default, tensors are created on CPU
# we move tensors to GPU using `.to` method after checking for availability
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print("First row: ", tensor[0])
print("First column: ", tensor[:, 0])
print("Last column: ", tensor[..., -1])
tensor[:, 1] = 0
print(tensor)

# joining tensors using cat/stack
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# arithmetic operations

# matrix multiplication, the following are all equivalent
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# element wise product, the following are all equivalent
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# in-place operations
# operators that store the result into the operand immediately
print(tensor, "\n")
tensor.add_(5)
print(tensor, "\n")

# Bridge with Numpy

# tensor to numpy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# change in tensor reflects in numpy array
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
