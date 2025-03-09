import torch

print("MPS est disponible :", torch.backends.mps.is_available())
print("MPS est supporté :", torch.backends.mps.is_built())