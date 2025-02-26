import numpy as np
import pdb
import torch
import torch
from pyquaternion import Quaternion
def quaternion_rotate_vector(quat,v_torch):

    q_conj = quat * torch.tensor([1, -1, -1, -1], dtype=quat.dtype, device=quat.device) # Conjugué du quaternion
    v_rotated_quat = quaternion_multiply(quaternion_multiply(quat, v_torch), q_conj)
    return  v_rotated_quat[1:].clone().detach()


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    stack_tensor = torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])
    return stack_tensor


quat_pyquat = Quaternion(0.5, 0.5, -0.5, -0.5) # w, x, y z
quat_array= np.array([quat_pyquat[0],quat_pyquat[1],quat_pyquat[2],quat_pyquat[3]])
quat_array_torch=torch.tensor(quat_array)


v = np.array([0, 0, 0, 1])  # mettr un zero devant
v_torch = torch.from_numpy(v)

ref_X_LinkFrame_in_wf =quaternion_rotate_vector(quat_array_torch,v_torch)
ref_Y_LinkFrame_in_wf =quaternion_rotate_vector(quat_array_torch,v_torch)
ref_Z_LinkFrame_in_wf = quaternion_rotate_vector(quat_array_torch, v_torch)

quat_pyquat_tensor =  torch.tensor([[ 0.5002,  0.4998, -0.5004, -0.4996],
        [ 0.5000,  0.5000, -0.5000, -0.5000],
        [ 0.5000,  0.5000, -0.5000, -0.5000]])



batched_function = torch.vmap(quaternion_rotate_vector, in_dims=(0,None))
result = batched_function(quat_pyquat_tensor,v_torch)
pdb.set_trace()



