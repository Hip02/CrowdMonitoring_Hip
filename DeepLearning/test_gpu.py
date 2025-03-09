import torch

print("CUDA est disponible :", torch.cuda.is_available())
print("CUDA est supporté :", torch.cuda.is_built())