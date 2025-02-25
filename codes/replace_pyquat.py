import numpy as np
import pdb
from pyquaternion import Quaternion
def quaternion_rotate_vector(q, v):
    q_conj = q * np.array([1, -1, -1, -1])  # Conjugué du quaternion
    v_quat = np.concatenate(([0], v))  # Vecteur en quaternion [0, x, y, z]

    v_rotated_quat = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)

    return v_rotated_quat[1:]


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

quat_pyquat = Quaternion(0.5, 0.5, -0.5, -0.5) # w, x, y z


quat_array= np.array([quat_pyquat[0],quat_pyquat[1],quat_pyquat[2],quat_pyquat[3]])


ref_axis_end_x = np.array([1, 0, 0])
ref_axis_end_y = np.array([0, 1, 0])
ref_axis_end_z = np.array([0, 0, 1])

ref_X_LinkFrame_in_wf =quaternion_rotate_vector(quat_array,ref_axis_end_x)
ref_Y_LinkFrame_in_wf =quaternion_rotate_vector(quat_array,ref_axis_end_y)
ref_Z_LinkFrame_in_wf = quaternion_rotate_vector(quat_array,ref_axis_end_z)

pdb.set_trace()


ref_axis_orthogonal_to_active_direction_local_frame = np.array([0, 0, 1])
ref_axis_orthogonal2_to_active_direction_local_frame = np.array([1, 0, 0])
ref_axis_active_direction_local_frame = np.array([0, 1, 0])

ref_axis_orthogonal_to_active_direction_world_frame =  quaternion_rotate_vector(quat_array, ref_axis_orthogonal_to_active_direction_local_frame)
ref_axis_orthogonal2_to_active_direction_world_frame = quaternion_rotate_vector(quat_array, ref_axis_orthogonal2_to_active_direction_local_frame)
ref_axis_active_direction_world_frame = quaternion_rotate_vector(quat_array, ref_axis_active_direction_local_frame)


pdb.set_trace()
