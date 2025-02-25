import pdb
import numpy as np
import torch
import torch

"""
ref_axis_orthogonal_to_active_direction_world_frame = quat_pyquat.rotate(
            ref_axis_orthogonal_to_active_direction_local_frame)
"""
tensor = torch.tensor([[1, 1, 1], [3, 3, 3]], dtype=torch.float32)
tensor = torch.tensor([[ 1,  1, 1, 1],
        [ 0.5000,  0.5000, 0.5000, 0.5000],
        [ 4,  4, 4, 4]])
ref_axis_orthogonal_to_active_direction_local_frame = np.array([0, 0, 1])


# Fonction personnalisée (ex: moyenne des carrés)
def custom_function(x):
    return 2*x

# Utilisation de torch.vmap (PyTorch 2.0+)
result = torch.vmap(custom_function, in_dims=0)(tensor)

print(result)  # tensor([ 4.6667, 41.0000])

pdb.set_trace()