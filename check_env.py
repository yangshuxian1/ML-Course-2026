import torch 
print(f"PyTorch version: {torch.__version__}") 
print(f"CUDA available: {torch.cuda.is_available()}") 
x = torch.rand(5, 3) 
print("Random tensor:\n", x) 
