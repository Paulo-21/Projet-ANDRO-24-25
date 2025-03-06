import pdb

import torch


start_angle =torch.tensor([[1],
        [2],
        [ 3],
        [ 4]], device='cuda:0')


ref_vect_for_arc = torch.tensor([[ 1, 2,  3],
        [ 10, 10,  10],
        [ 4, 4,  4],
        [ 0.0000, 2,  2]], device='cuda:0')

def rotation_matrix_theta_around_axisU(start_angle, ref_vect_for_arc):
        ux=ref_vect_for_arc[0]
        uy=ref_vect_for_arc[1]
        uz=ref_vect_for_arc[2]
        print("ux",ux)
        return  ux*10,uy*100,uz*100


batched_function = torch.vmap(rotation_matrix_theta_around_axisU, in_dims=(0,0))
result = batched_function(start_angle, ref_vect_for_arc)
pdb.set_trace()