import pdb

import torch
from functorch import vmap  # functorch est intégré dans torch 2.0+

A = torch.tensor([[-2.7786],
                  [ 1.9710],
                  [-2.7906]], device='cuda:0')

B = torch.tensor([[-1.9932],
                  [ 2.7564],
                  [-2.0052]], device='cuda:0')

n_step = 5  # Doit être un int

# Définir une fonction qui applique torch.linspace élément par élément
def linspace_fn(a_input, b_input):
    pdb.set_trace()
    return torch.linspace(a_input.item(), b_input.item(), n_step, device='cuda:0')

# Appliquer vmap pour vectoriser
batched_function = torch.vmap(linspace_fn, in_dims=(1,1))
result = batched_function(A,B)

print(result)