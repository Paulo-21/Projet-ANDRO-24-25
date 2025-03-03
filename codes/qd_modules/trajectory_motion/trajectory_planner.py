import pdb
import sys
import os
import time
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import csv
import json
from numpy import linalg as LA
import torch
import genesis as gs
from ..actions_achitecture_module.actions_architectures import *
from ..utils_bootstrap.bootstrap_related import *
from ..actions_achitecture_module.genesis_actions_architecture import *
from pyquaternion import Quaternion
import math
from ..trajectory_motion.colors_const import *
def matmul_torch(rotation_matrix_torch,ref_vect_for_arc):
    return torch.matmul(rotation_matrix_torch, ref_vect_for_arc)

def compute_tensor(theta, vectors):
    ux, uy, uz = vectors[:, 0], vectors[:, 1], vectors[:, 2]  # Décomposer les coordonnées
    theta_squeezed = theta.squeeze(-1)
    c = torch.cos(theta_squeezed)
    s = torch.sin(theta_squeezed)

    matrix = torch.stack(
        [torch.stack([ux ** 2 + (1 - ux ** 2) * c, ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s], dim=-1),
         torch.stack([ux * uy * (1 - c) + uz * s, uy ** 2 + (1 - uy ** 2) * c, uy * uz * (1 - c) - ux * s], dim=-1),
         torch.stack([ux * uz * (1 - c) - uy * s, uy * uz * (1 - c) + ux * s, uz ** 2 + (1 - uz ** 2) * c], dim=-1)],
                     dim=1)
    return matrix

def cross_torch(a,b):
    batched_function = torch.vmap(torch_cross_multi_dim, in_dims=(0, 0))
    result = batched_function(a, b)
    return result

def dot_torch(a,b):
    batched_function = torch.vmap(torch_dot_multi_dim, in_dims=(0, 0))
    result = batched_function(a, b)
    return result


def dot_torch_capsule(a, b):
    result = dot_torch(a, b)
    to_mult = result.unsqueeze(1)
    final = to_mult * b
    return final
def torch_dot_multi_dim(a,b):
    return  torch.dot(a,b)
def torch_cross_multi_dim(a,b):
    return torch.cross(a, b)

def torch_quaternion_rotate_vector(quat,v_torch):
    q_conj = quat * torch.tensor([1, -1, -1, -1], dtype=quat.dtype, device=quat.device) # Conjugué du quaternion
    v_rotated_quat = torch_quaternion_multiply(torch_quaternion_multiply(quat, v_torch), q_conj)
    return  v_rotated_quat[1:].clone().detach()


def torch_quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    stack_tensor = torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])
    return stack_tensor

def rot_mtx_fct( quat_pyquat_simple_array):
    rotation_matrix = quat_pyquat_simple_array.rotation_matrix
    return rotation_matrix

def quaternion_product(child_link_quat_pyquat,parent_link_quat_pyquat, child_link_pose_wf):
        child_link_quat_pyquat =Quaternion(child_link_quat_pyquat)
        parent_link_quat_pyquat = Quaternion(parent_link_quat_pyquat)
        child_link_pose_wf = child_link_pose_wf

        quat_pyquat = child_link_quat_pyquat * parent_link_quat_pyquat
        rotation_matrix = quat_pyquat.rotation_matrix

        tx, ty, tz = child_link_pose_wf[0].item(), child_link_pose_wf[1].item(), child_link_pose_wf[
            2].item()
        T = [tx, ty, tz]
        O_link_frame_wf = np.array(T)
        matrice_homogene = concat_rotation_t_matrix(rotation_matrix=rotation_matrix, T=T)
        quat_pyquat_torch = torch.Tensor(quat_pyquat.q).to("cuda")
        rotation_matrix_torch = torch.Tensor(rotation_matrix).to("cuda")
        matrice_homogene_torch = torch.Tensor(matrice_homogene).to("cuda")
        O_link_frame_wf_torch = torch.Tensor(O_link_frame_wf).to("cuda")

        return quat_pyquat_torch,rotation_matrix_torch, matrice_homogene_torch, O_link_frame_wf_torch

def concat_rotation_t_matrix(rotation_matrix, T):
    tx, ty, tz = T[0], T[1], T[2]
    matrice_homogene = np.array([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], tx],
                                 [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], ty],
                                 [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], tz],
                                 [0, 0, 0, 1]],
                                )
    return matrice_homogene

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


class Trajectory_planner():
    def __init__(self):
        self.test=None

    def generate_arc(self,ref_vect_for_arc, u_vect, stop_angle=30, start_angle=0, num_points=100, center_point=0,multi_thread=None, increment_total_in_deg=2
            ):
        if multi_thread=="GPU_simple":
            angles_rad = np.linspace(np.radians(start_angle), np.radians(stop_angle), num_points)
            x = []
            y = []
            z = []
            for angle_rad in angles_rad:
                rotation_matrix = self.rotation_matrix_theta_around_axisU(u=u_vect, theta_rad=angle_rad)
                start_vect = np.dot(rotation_matrix, ref_vect_for_arc)

                trajectory_point_in_wrold_space = center_point + start_vect
                [x_coordinate,y_coordinate,z_coordinate]=trajectory_point_in_wrold_space
                x.append(x_coordinate)
                y.append(y_coordinate)
                z.append(z_coordinate)
            x=np.array(x)
            y=np.array(y)
            z=np.array(z)

        elif multi_thread=="GPU_parallel" :

            increment_total_in_rad = increment_total_in_deg *np.pi/180
            deg_for_one_step = increment_total_in_rad / num_points
            start_angle_rad = torch.deg2rad(start_angle)
            x = torch.empty(0,  len(start_angle), device="cuda")
            y = torch.empty(0,  len(start_angle), device="cuda")
            z = torch.empty(0,  len(start_angle), device="cuda")

            for theta_column_item in range(num_points):
                theta_column = start_angle_rad + theta_column_item*deg_for_one_step
                rotation_matrix_torch = compute_tensor(theta_column, u_vect)
                batched_function = torch.vmap(matmul_torch, in_dims=(0, 0))
                start_vect_torch = batched_function(rotation_matrix_torch,ref_vect_for_arc)
                trajectory_point_in_wrold_space = center_point + start_vect_torch
                x_one = trajectory_point_in_wrold_space[:,0]
                y_one = trajectory_point_in_wrold_space[:,1]
                z_one = trajectory_point_in_wrold_space[:,2]
                x = torch.cat((x, x_one.unsqueeze(0)), dim=0)
                y = torch.cat((y, y_one.unsqueeze(0)), dim=0)
                z = torch.cat((z, z_one.unsqueeze(0)), dim=0)

        else:
            raise ValueError('there is a pb in multithread name')

        return x, y, z

    def rotation_matrix_theta_around_axisU(self,u,theta_rad):
        ux = u[0]
        uy= u[1]
        uz=u[2]
        c=np.cos(theta_rad)
        s=np.sin(theta_rad)
        rotation_matrix = np.array([
            [ux**2+(1-ux**2)*c, ux*uy*(1-c)-uz*s, ux*uz*(1-c) + uy*s],
            [ux*uy*(1-c)+uz*s, uy**2+(1-uy**2)*c, uy*uz*(1-c)-ux*s],
            [ux*uz*(1-c)-uy*s, uy*uz*(1-c)+ux*s, uz**2+(1-uz**2)*c],
        ])
        return rotation_matrix

    def affect_xux_uy_uz(self,tensor_one_dim):
        ux = tensor_one_dim[0]
        uy = tensor_one_dim[1]
        uz = tensor_one_dim[2]
        pdb.set_trace()
        return ux,uy,uz

    def rotation_matrix_theta_around_axisU_torch(self, u_vect, theta_rad):
        ux = u_vect[0]
        uy = u_vect[1]
        uz = u_vect[2]
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        rotation_matrix = np.array([
            [ux ** 2 + (1 - ux ** 2) * c, ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [ux * uy * (1 - c) + uz * s, uy ** 2 + (1 - uy ** 2) * c, uy * uz * (1 - c) - ux * s],
            [ux * uz * (1 - c) - uy * s, uy * uz * (1 - c) + ux * s, uz ** 2 + (1 - uz ** 2) * c],
        ])
        return rotation_matrix



    def get_rotation_mtx(self, multi_thread, child_link_quat_pyquat, parent_link_quat_pyquat, child_link_pose_wf):
        if multi_thread=="GPU_simple":
            quat_pyquat = child_link_quat_pyquat * parent_link_quat_pyquat
            rotation_matrix = quat_pyquat.rotation_matrix
            tx, ty, tz = child_link_pose_wf[0].item(), child_link_pose_wf[1].item(), child_link_pose_wf[2].item()
            O_link_frame_wf = np.array([tx, ty, tz])
            T = [tx, ty, tz]
            matrice_homogene = concat_rotation_t_matrix(rotation_matrix=rotation_matrix, T=T)

        if multi_thread == "GPU_parallel":
            parent_link_quat_pyquat_array = parent_link_quat_pyquat
            child_link_quat_pyquat_array = child_link_quat_pyquat
            child_link_pose_wf_array = child_link_pose_wf
            quat_pyquat_torch_concat = torch.empty(0, 4, device="cuda")
            rotation_matrix_torch_concat =  torch.empty(0, 3,3, device="cuda")
            matrice_homogene_torch_concat = torch.empty(0, 4,4, device="cuda")
            O_link_frame_wf_concat =torch.empty(0, 3, device="cuda")

            for i in range(len(child_link_quat_pyquat)):
                parent_link_quat_pyquat = parent_link_quat_pyquat_array[i].cpu().numpy()
                child_link_quat_pyquat = child_link_quat_pyquat_array[i].cpu().numpy()
                child_link_pose_wf = child_link_pose_wf_array[i].cpu().numpy()

                quat_pyquat_torch, rotation_matrix_torch, matrice_homogene_torch, O_link_frame_wf_torch = quaternion_product(
                    child_link_quat_pyquat=child_link_quat_pyquat,
                    parent_link_quat_pyquat=parent_link_quat_pyquat,
                    child_link_pose_wf=child_link_pose_wf)
                quat_pyquat_torch_concat =  torch.cat((quat_pyquat_torch_concat, quat_pyquat_torch.unsqueeze(0)), dim=0) #torch.stack((quat_pyquat_torch_concat, quat_pyquat_torch ),dim=0)
                rotation_matrix_torch_concat = torch.cat((rotation_matrix_torch_concat, rotation_matrix_torch.unsqueeze(0)), dim=0)  # torch.stack((rotation_matrix_torch_concat, rotation_matrix_torch), dim=0)
                matrice_homogene_torch_concat = torch.cat((matrice_homogene_torch_concat, matrice_homogene_torch.unsqueeze(0)), dim=0)  # torch.stack ((matrice_homogene_torch_concat, matrice_homogene_torch), dim=0)
                O_link_frame_wf_concat = torch.cat((O_link_frame_wf_concat, O_link_frame_wf_torch.unsqueeze(0)), dim=0)

            matrice_homogene = matrice_homogene_torch_concat
            quat_pyquat = quat_pyquat_torch_concat
            O_link_frame_wf = O_link_frame_wf_concat
        return matrice_homogene, quat_pyquat, O_link_frame_wf



    def apply_rotation_arround_articulation(self,sim_scene, nbr_item_joint_studied=3, multi_thread=None,  geometric_debug=True,direction="positive"):

        display_debug_geometry = False
        display_force_array = False
        trajectory_geometric_debug = True #display trajectroy motion
        nbr_step_for_one_command = 100 # sans debug100
        nbr_step_for_complient_control = 1000 #1000 sans debug
        num_points = 300 #10 for debug but 100 for normal
        delta_deg_angle = 50


        small_force_appliend_on_fingers = -0.02
        hard_force_appliend_on_fingers = -1.5 #-1.5#0-1.5 pour le frigot
        sim_scene.command_open_fingers(multi_thread)

        for i in range(nbr_step_for_complient_control):
            print("i = ", i)
            sim_scene.apply_force_on_robot(small_force_appliend_on_fingers, multi_thread)
            sim_scene.command_remain_origin(multi_thread)
            if display_force_array:
                contact_arrows = sim_scene.robot.get_links_net_contact_force()
                contact_left_arrow = contact_arrows[1].cpu().numpy()
                contact_right_arrow = contact_arrows[2].cpu().numpy()
                print("contact_left_arrow", contact_left_arrow)
                print("contact_right_arrow", contact_right_arrow)
                if display_debug_geometry:
                    sim_scene.scene.draw_debug_arrow([0, 0, 0], 10 * contact_left_arrow, radius=0.0006, color=vert)
                    sim_scene.scene.draw_debug_arrow([0, 0, 0], 10 * contact_right_arrow, radius=0.0006, color=rouge)
            sim_scene.scene.step()

        print("End of applying small control force")
        init_action_object_joint_values = sim_scene.get_list_of_object_joint_values(multi_thread)

        child_link_quat_pyquat, parent_link_quat_pyquat, child_link_pose_wf,dof_motion_angle_lf = sim_scene.get_parent_child_info(multi_thread,nbr_item_joint_studied)
        matrice_homogene, quat_pyquat,O_link_frame_wf = self.get_rotation_mtx( multi_thread, child_link_quat_pyquat, parent_link_quat_pyquat, child_link_pose_wf)

        B = np.absolute(dof_motion_angle_lf) #si valeur absolue est - ou plsu va changer
        value_active_axis_local_ref = np.argmax(B)
        ref_axis_end_z = np.array([0, 0, 1])
        ref_axis_end_x = np.array([1, 0, 0])
        ref_axis_end_y = np.array([0, 1, 0])

        if value_active_axis_local_ref == 0:
            active_axis_cartesian_local_ref = "x"
            ref_axis_orthogonal_to_active_direction_local_frame = ref_axis_end_z
            ref_axis_orthogonal2_to_active_direction_local_frame = ref_axis_end_y
            ref_axis_active_direction_local_frame = np.array([1, 0, 0])
            rot2 = Quaternion(axis=np.array([1, 0, 0]), degrees=90)

        elif value_active_axis_local_ref == 1:
            active_axis_cartesian_local_ref = "y"
            ref_axis_orthogonal_to_active_direction_local_frame = ref_axis_end_z
            ref_axis_orthogonal2_to_active_direction_local_frame = ref_axis_end_x
            ref_axis_active_direction_local_frame = np.array([0, 1, 0])
            rot2 = Quaternion(axis=np.array([0, 1, 0]), degrees=0)

        elif value_active_axis_local_ref == 2:
            active_axis_cartesian_local_ref = "z"
            ref_axis_orthogonal_to_active_direction_local_frame = ref_axis_end_y
            ref_axis_orthogonal2_to_active_direction_local_frame = ref_axis_end_x
            ref_axis_active_direction_local_frame = np.array([0, 0, 1])
            rot2 = Quaternion(axis=np.array([0, 1, 0]), degrees=0)  # car joint 3 et 1
        else:
            raise Exception('erreur dans l affectation de l axe actif')

        if multi_thread=="GPU_simple":
            quat_array = np.array([quat_pyquat[0],quat_pyquat[1],quat_pyquat[2],quat_pyquat[3]])
            ref_axis_orthogonal_to_active_direction_world_frame = quaternion_rotate_vector(quat_array,ref_axis_orthogonal_to_active_direction_local_frame)
            ref_axis_orthogonal2_to_active_direction_world_frame= quaternion_rotate_vector(quat_array,ref_axis_orthogonal2_to_active_direction_local_frame)
            ref_axis_active_direction_world_frame = quaternion_rotate_vector(quat_array,ref_axis_active_direction_local_frame)


            ref_X_LinkFrame_in_wf = quaternion_rotate_vector(quat_array,ref_axis_end_x)
            ref_Y_LinkFrame_in_wf = quaternion_rotate_vector(quat_array,ref_axis_end_y)
            ref_Z_LinkFrame_in_wf = quaternion_rotate_vector(quat_array,ref_axis_end_z)

            O_link_frame_wf_debug = O_link_frame_wf
            ref_X_LinkFrame_in_wf_debug = ref_X_LinkFrame_in_wf
            ref_Y_LinkFrame_in_wf_debug = ref_Y_LinkFrame_in_wf
            ref_Z_LinkFrame_in_wf_debug = ref_Z_LinkFrame_in_wf
        else:
            ref_axis_end_x_torch_v = np.concatenate(([0], ref_axis_end_x))
            ref_axis_end_y_torch_v = np.concatenate(([0], ref_axis_end_y))
            ref_axis_end_z_torch_v = np.concatenate(([0], ref_axis_end_z))

            ref_axis_orthogonal_to_active_direction_local_frame_torch_v = np.concatenate(([0], ref_axis_orthogonal_to_active_direction_local_frame))
            ref_axis_orthogonal2_to_active_direction_local_frame_torch_v = np.concatenate(([0], ref_axis_orthogonal2_to_active_direction_local_frame))
            ref_axis_active_direction_world_frame_torch_v = np.concatenate(([0], ref_axis_active_direction_local_frame))

            ref_axis_end_x_torch = torch.from_numpy(ref_axis_end_x_torch_v)
            ref_axis_end_y_torch = torch.from_numpy(ref_axis_end_y_torch_v)
            ref_axis_end_z_torch = torch.from_numpy(ref_axis_end_z_torch_v)

            ref_axis_orthogonal_to_active_direction_local_frame_torch = torch.from_numpy(ref_axis_orthogonal_to_active_direction_local_frame_torch_v)
            ref_axis_orthogonal2_to_active_direction_local_frame_torch = torch.from_numpy(ref_axis_orthogonal2_to_active_direction_local_frame_torch_v)
            ref_axis_active_direction_world_frame_torch = torch.from_numpy(ref_axis_active_direction_world_frame_torch_v)

            batched_function = torch.vmap(torch_quaternion_rotate_vector, in_dims=(0, None))
            ref_X_LinkFrame_in_wf_torch = batched_function(quat_pyquat, ref_axis_end_x_torch)
            ref_Y_LinkFrame_in_wf_torch = batched_function(quat_pyquat,ref_axis_end_y_torch)
            ref_Z_LinkFrame_in_wf_torch = batched_function(quat_pyquat, ref_axis_end_z_torch)

            ref_axis_orthogonal_to_active_direction_world_frame = batched_function(quat_pyquat, ref_axis_orthogonal_to_active_direction_local_frame_torch)
            ref_axis_orthogonal2_to_active_direction_world_frame = batched_function(quat_pyquat, ref_axis_orthogonal2_to_active_direction_local_frame_torch)
            ref_axis_active_direction_world_frame = batched_function(quat_pyquat, ref_axis_active_direction_world_frame_torch)
            O_link_frame_wf_debug = O_link_frame_wf.cpu()[0]
            ref_X_LinkFrame_in_wf_debug =ref_X_LinkFrame_in_wf_torch.cpu()[0]
            ref_Y_LinkFrame_in_wf_debug = ref_Y_LinkFrame_in_wf_torch.cpu()[0]
            ref_Z_LinkFrame_in_wf_debug = ref_Z_LinkFrame_in_wf_torch.cpu()[0]
        if display_debug_geometry:
            sim_scene.scene.draw_debug_arrow(O_link_frame_wf_debug, 0.3 * ref_X_LinkFrame_in_wf_debug, radius=0.01,
                                             color=(1, 0, 0, 0.5))
            sim_scene.scene.draw_debug_arrow(O_link_frame_wf_debug, 0.3 * ref_Y_LinkFrame_in_wf_debug, radius=0.01,
                                             color=(0, 1, 0, 0.5))
            sim_scene.scene.draw_debug_arrow(O_link_frame_wf_debug, 0.3 * ref_Z_LinkFrame_in_wf_debug, radius=0.01,
                                             color=(0, 0, 1.0, 0.5))
            sim_scene.scene.draw_debug_sphere(O_link_frame_wf_debug, radius=0.01, color=(1.0, 0.0, 0.0, 0.5))

        O_point_wf = O_link_frame_wf
        if multi_thread=="GPU_simple":
            Mprime_point_wf = sim_scene.robot.get_pos().cpu().numpy()
        elif multi_thread=="GPU_parallel" :
            Mprime_point_wf = sim_scene.robot.get_pos()
        else :
            pass

        R_point_wf = O_point_wf + ref_axis_orthogonal_to_active_direction_world_frame
        R2_point_wf = O_point_wf + ref_axis_orthogonal2_to_active_direction_world_frame

        A_point_wf = O_point_wf + ref_axis_active_direction_world_frame
        OA_vect = A_point_wf - O_point_wf

        OR_vect_wf = R_point_wf - O_point_wf
        OR2_vect_wf = R2_point_wf - O_point_wf
        OMprime_vect = Mprime_point_wf - O_point_wf
        if multi_thread=="GPU_simple":
            O_point_wf_debug = O_point_wf
            OMprime_vect_debug = OMprime_vect
        elif multi_thread=="GPU_parallel":
            O_point_wf_debug =O_point_wf.cpu()[0].numpy()
            OMprime_vect_debug =  OMprime_vect.cpu()[0].numpy()
        if display_debug_geometry:
            sim_scene.scene.draw_debug_arrow(O_point_wf_debug, OMprime_vect_debug, radius=0.006, color=jaune)
        # Projete dans le repere de l articulation
        if multi_thread=="GPU_simple":
            OMprime_on_OA_vect = np.dot(OMprime_vect, OA_vect) * OA_vect
            OMprime_on_R2_vect = np.dot(OMprime_vect, OR2_vect_wf) * OR2_vect_wf
            OMprime_on_R_vect = np.dot(OMprime_vect, OR_vect_wf) * OR_vect_wf
        if multi_thread=="GPU_parallel":
            OMprime_on_OA_vect = dot_torch_capsule(a=OMprime_vect,b=OA_vect)
            OMprime_on_R2_vect = dot_torch_capsule(a=OMprime_vect, b=OR2_vect_wf)
            OMprime_on_R_vect = dot_torch_capsule(a=OMprime_vect, b=OR_vect_wf)

        Mprime_on_OA_point= O_point_wf + OMprime_on_OA_vect
        H_point = Mprime_on_OA_point
        HMprime_vect  = Mprime_point_wf - H_point
        Mprime_on_OR2_point =  O_point_wf + OMprime_on_R2_vect
        Mprime_on_OR_point = O_point_wf + OMprime_on_R_vect

        if multi_thread == "GPU_simple":
            Mprime_on_OA_point_debug = Mprime_on_OA_point
            Mprime_on_OR2_point_debug = Mprime_on_OR2_point
            Mprime_on_OR_point_debug = Mprime_on_OR_point
            R2_point_wf_debug = R2_point_wf
            R_point_wf_debug = R_point_wf
            H_point_debug = H_point
            Mprime_point_wf_debug = Mprime_point_wf
            OMprime_on_OA_vect_debug = OMprime_on_OA_vect
            OMprime_on_R2_vect_debug = OMprime_on_R2_vect
            HMprime_vect_debug = HMprime_vect
            HMprime_vect_debug = HMprime_vect
            OMprime_on_R2_vect_debug = OMprime_on_R2_vect
            OR2_vect_wf_debug =  OR2_vect_wf
        elif multi_thread == "GPU_parallel":
            Mprime_on_OA_point_debug = Mprime_on_OA_point[0].cpu().numpy()
            Mprime_on_OR2_point_debug = Mprime_on_OR2_point[0].cpu().numpy()
            Mprime_on_OR_point_debug = Mprime_on_OR_point[0].cpu().numpy()
            R2_point_wf_debug = R2_point_wf[0].cpu().numpy()
            R_point_wf_debug = R_point_wf[0].cpu().numpy()
            H_point_debug = H_point[0].cpu().numpy()
            Mprime_point_wf_debug = Mprime_point_wf[0].cpu().numpy()
            OMprime_on_OA_vect_debug = OMprime_on_OA_vect[0].cpu().numpy()
            OMprime_on_R2_vect_debug = OMprime_on_R2_vect[0].cpu().numpy()
            HMprime_vect_debug = HMprime_vect[0].cpu().numpy()
            HMprime_vect_debug = HMprime_vect[0].cpu().numpy()
            OMprime_on_R2_vect_debug = OMprime_on_R2_vect[0].cpu().numpy()
            OR2_vect_wf_debug =  OR2_vect_wf[0].cpu().numpy()
        else :
            raise ValueError("pb multi thread")
        if display_debug_geometry :
            sim_scene.scene.draw_debug_sphere(Mprime_on_OA_point_debug, radius=0.01, color=mauve)
            sim_scene.scene.draw_debug_sphere(Mprime_on_OR2_point_debug, radius=0.01, color=mauve)
            sim_scene.scene.draw_debug_sphere(Mprime_on_OR_point_debug, radius=0.01, color=mauve)
            sim_scene.scene.draw_debug_sphere(R2_point_wf_debug, radius=0.01, color=mauve)
            sim_scene.scene.draw_debug_sphere(R_point_wf_debug, radius=0.01, color=mauve)

            sim_scene.scene.draw_debug_sphere(H_point_debug, radius=0.01, color=mauve)
            sim_scene.scene.draw_debug_sphere(Mprime_point_wf_debug, radius=0.01, color=mauve)

            sim_scene.scene.draw_debug_arrow(O_point_wf_debug, OMprime_on_OA_vect_debug, radius=0.006, color=mauve)
            sim_scene.scene.draw_debug_arrow(O_point_wf_debug, OMprime_on_R2_vect_debug, radius=0.006, color=mauve)
            sim_scene.scene.draw_debug_arrow(H_point_debug, HMprime_vect_debug, radius=0.006, color=mauve)
            sim_scene.scene.draw_debug_arrow(O_point_wf_debug, HMprime_vect_debug, radius=0.006, color=mauve)
            sim_scene.scene.draw_debug_arrow(Mprime_on_OR_point_debug, OMprime_on_R2_vect_debug, radius=0.006, color=mauve)
        if multi_thread=="GPU_simple":
            nom_radius_around_axis = np.linalg.norm(HMprime_vect)
            vect_reference_radius_theta_zero = nom_radius_around_axis * OR2_vect_wf
            theta_rad = np.arccos(
                np.dot(OR2_vect_wf, HMprime_vect) / (np.linalg.norm(OR2_vect_wf) * np.linalg.norm(HMprime_vect)))


        elif multi_thread=="GPU_parallel":
            nom_radius_around_axis = torch.norm(HMprime_vect, dim=1)
            vect_reference_radius_theta_zero = nom_radius_around_axis.unsqueeze(1) * OR2_vect_wf
            A = dot_torch(a=OR2_vect_wf, b=HMprime_vect).unsqueeze(1)
            B = torch.norm(OR2_vect_wf, dim=1).unsqueeze(1)
            C =  torch.norm(HMprime_vect, dim=1).unsqueeze(1)
            D = B * C
            E  = A/D
            theta_rad = torch.acos(E)

        if multi_thread=="GPU_simple":
            rot_vect_in_wf = np.cross(OR2_vect_wf,HMprime_vect)
            value_rot_axis_local_ref = np.argmax(np.abs(rot_vect_in_wf))
            sign_rot = np.sign(rot_vect_in_wf[value_rot_axis_local_ref])

            value_active_axis_local_ref = np.argmax(np.abs(OA_vect))
            sign_active_axis = np.sign(OA_vect[value_active_axis_local_ref])
            if sign_active_axis != sign_rot:
                theta_rad = -theta_rad
        else :
            rot_vect_in_wf = cross_torch(OR2_vect_wf, HMprime_vect)
            A =  torch.abs(rot_vect_in_wf)
            value_rot_axis_local_ref = torch.argmax(A,dim=1)
            B = rot_vect_in_wf[torch.arange(rot_vect_in_wf.size(0)), value_rot_axis_local_ref]
            sign_rot = torch.sign(B)

            A = torch.abs(OA_vect)
            value_active_axis_local_ref =  torch.argmax(A,dim=1)
            B = OA_vect[torch.arange(OA_vect.size(0)), value_active_axis_local_ref]
            sign_active_axis = torch.sign(B)

            mask = sign_active_axis != sign_rot
            theta_rad[mask] = -theta_rad[mask]

        if display_debug_geometry:
            sim_scene.scene.draw_debug_arrow(H_point_debug, 3*HMprime_vect_debug, radius=0.0006, color=bordeaux)
            sim_scene.scene.draw_debug_arrow(O_point_wf_debug, 3*OMprime_on_R2_vect_debug, radius=0.0006, color=bordeaux)
            sim_scene.scene.draw_debug_arrow(H_point_debug,  HMprime_vect_debug, radius=0.003, color=bordeaux)
            sim_scene.scene.draw_debug_arrow(O_point_wf_debug,  OR2_vect_wf_debug, radius=0.003, color=bordeaux)
            #sim_scene.scene.draw_debug_arrow(O_point_wf, 3 * OMprime_on_R2_vect, radius=0.006, color=bordeaux)


        start_angle_deg = theta_rad * 180 / np.pi

        increment_total_in_deg = delta_deg_angle

        if direction=="positive":
            stop_angle_deg = start_angle_deg + delta_deg_angle
        elif direction=="negative":
            stop_angle_deg = start_angle_deg - delta_deg_angle
        else:
            raise TypeError('No good direction given')

        stop_angle_rad = stop_angle_deg*np.pi/180
        if multi_thread=="GPU_simple":
            theta_rad_debug=theta_rad
            OA_vect_debug=OA_vect
            stop_angle_rad_debug=stop_angle_rad
            vect_reference_radius_theta_zero_debug=vect_reference_radius_theta_zero
        else:
            theta_rad_debug=theta_rad[0].cpu().numpy()[0]
            OA_vect_debug=OA_vect[0].cpu().numpy()
            stop_angle_rad_debug=stop_angle_rad[0].cpu().numpy()[0]
            vect_reference_radius_theta_zero_debug=vect_reference_radius_theta_zero[0].cpu().numpy()

        if display_debug_geometry:
            test_lispace_debug = np.linspace(0,theta_rad_debug,20)
            for theta_rad_debug in test_lispace_debug:
                rotation_matrix_start_debug = self.rotation_matrix_theta_around_axisU(u=OA_vect_debug, theta_rad=theta_rad_debug)
                rotation_matrix_stop_debug = self.rotation_matrix_theta_around_axisU(u=OA_vect_debug, theta_rad=stop_angle_rad_debug)
                start_vect_debug  = np.dot(rotation_matrix_start_debug, vect_reference_radius_theta_zero_debug)
                stop_vect_debug = np.dot(rotation_matrix_stop_debug, vect_reference_radius_theta_zero_debug)

                sim_scene.scene.draw_debug_arrow(O_point_wf_debug, start_vect_debug, radius=0.001, color=vert_canard)
                sim_scene.scene.draw_debug_arrow(O_point_wf_debug, stop_vect_debug, radius=0.001, color=vert_canard)


        x_list, y_list, z_list = self.generate_arc(
                                              ref_vect_for_arc=vect_reference_radius_theta_zero,
                                              u_vect=OA_vect,
                                              stop_angle=stop_angle_deg,
                                              start_angle=start_angle_deg,
                                              num_points=num_points,
                                              center_point=H_point,
                                                   multi_thread=multi_thread,
                                                   increment_total_in_deg=increment_total_in_deg)
        if multi_thread=="GPU_simple":
            x_list_debug = x_list
            y_list_debug = y_list
            z_list_debug = z_list
        else :
            x_list_debug = x_list[:,0].cpu().numpy()
            y_list_debug = y_list[:,0].cpu().numpy()
            z_list_debug = z_list[:,0].cpu().numpy()
        if trajectory_geometric_debug:
            for i in range(num_points):
                x = x_list_debug[i]
                y = y_list_debug[i]
                z = z_list_debug[i]
                if i==0:
                    if display_debug_geometry:
                        sim_scene.scene.draw_debug_sphere(np.array([x, y, z]), radius=0.02, color=[0,1,0,1])
                    else :
                        pass
                elif i==(num_points-1):
                    if display_debug_geometry:
                        sim_scene.scene.draw_debug_sphere(np.array([x, y, z]), radius=0.02, color=[1, 0, 0, 1])
                    else:
                        pass
                else:
                    if display_debug_geometry:
                        sim_scene.scene.draw_debug_sphere(np.array([x, y, z]), radius=0.01, color=jaune)
                    else:
                        pass
        n_waypoint = 0

        n_step = num_points

        sim_scene.set_kp_parameters(multi_thread)
        for i in range(n_step):
            sim_scene.command_pos_of_robot(x_list, y_list, z_list, n_waypoint, multi_thread)
            for _ in range(nbr_step_for_one_command):
                sim_scene.command_pos_of_object_static(multi_thread)
                sim_scene.apply_force_on_robot(hard_force_appliend_on_fingers, multi_thread)
                sim_scene.scene.step()
            n_waypoint += 1
        difference_init_end_action_joint_value = sim_scene.how_actionable_grasp(
            multi_thread=multi_thread,
            init_action_object_joint_values=init_action_object_joint_values)
        # sim_scene.scene.clear_debug_objects()
        pdb.set_trace()

        return difference_init_end_action_joint_value


    def rotate_around_axis(self,sim_scene):
        index_object_articulation = 2
        difference_init_end_action_joint_value = self.sim_scene.apply_rotation_arround_articulation(
            nbr_item_joint_studied=index_object_articulation)




    def from_pyquat_to_genesis(self,pyquat):
        genesis_quat = np.array([pyquat[3], pyquat[0], pyquat[1], pyquat[2]])
        return genesis_quat



