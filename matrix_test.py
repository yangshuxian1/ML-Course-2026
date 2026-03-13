import torch 
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) 
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]]) 
C = torch.matmul(A, B) 
print("Matrix A:\n", A) 
print("Matrix B:\n", B) 
print("Result C = A * B:\n", C) 
