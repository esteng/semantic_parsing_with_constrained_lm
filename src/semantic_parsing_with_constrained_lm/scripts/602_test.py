import torch


is_cuda = torch.cuda.is_available()

print(f"IS CUDA: {is_cuda}") 
