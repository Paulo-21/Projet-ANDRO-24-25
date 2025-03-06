import numpy as np
import pdb
import torch
import torch
from pyquaternion import Quaternion
def torch_dot_multi_dim(a,b):
    return  torch.dot(a,b)

BA_vect = torch.tensor([[0.0000e+00, 0.0000e+00, 1.0000e+00],
        [2.3848e-04, 0.0000e+00, 1.0000e+00],
        [0.0000e+00, 0.0000e+00, 1.0000e+00]], device='cuda:0')

BM_prime = torch.tensor([[-0.2335,  0.6148, -0.5152],
        [ 0.5924,  0.5238, -1.6584],
        [-0.0037, -0.4404, -1.4456]], device='cuda:0')



test1 = torch.tensor([ 1,  10, 100],device="cuda")

batched_function = torch.vmap(torch_dot_multi_dim, in_dims=(0,0))
result = batched_function(BM_prime, BA_vect)
to_mult = result.unsqueeze(1)
final = to_mult * BA_vect
pdb.set_trace()



