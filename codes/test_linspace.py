import torch

# Définition des bornes min (A) et max (B)
A = torch.tensor([[-2.7786],
                  [ 1.9710],
                  [-2.7906]], device='cuda:0')

B = torch.tensor([[-1.9932],
                  [ 2.7564],
                  [-2.0052]], device='cuda:0')

steps = 5  # Nombre de points intermédiaires

# S'assurer que les tenseurs sont de forme (N,)
A = A.squeeze(-1)
B = B.squeeze(-1)

# Générer le linspace pour chaque élément avec torch.vmap
batched_linspace = torch.vmap(lambda a, b: torch.linspace(a, b, steps, device='cuda:0'))

# Appliquer la vectorisation
result = batched_linspace(A, B)

print(result)  # Tenseur de forme (N, steps)
