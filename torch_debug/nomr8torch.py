import pdb

import torch
c = torch.tensor([[ 1, 2, 3], [1, 2, 3]] , dtype=torch.float)
torch.norm(c, dim=1)
pdb.set_trace()