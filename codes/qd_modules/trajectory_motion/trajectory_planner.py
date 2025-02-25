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
        matrice_homogene = concat_rotation_t_matrix(rotation_matrix=rotation_matrix, T=T)
        quat_pyquat_torch = torch.Tensor(quat_pyquat.q)
        rotation_matrix_torch = torch.Tensor(rotation_matrix)
        matrice_homogene_torch = torch.Tensor(matrice_homogene)

        return quat_pyquat_torch,rotation_matrix_torch, matrice_homogene_torch

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

    def generate_arc(self,sim_scene,ref_vect_for_arc, u, stop_angle=30, start_angle=0, num_points=100, center_point=0,
            ):
        angles_rad = np.linspace(np.radians(start_angle), np.radians(stop_angle), num_points)
        x = []
        y = []
        z = []
        for angle_rad in angles_rad:
            rotation_matrix = self.rotation_matrix_theta_around_axisU(u=u, theta_rad=angle_rad)
            start_vect = np.dot(rotation_matrix, ref_vect_for_arc)
            trajectory_point_in_wrold_space = center_point + start_vect
            [x_coordinate,y_coordinate,z_coordinate]=trajectory_point_in_wrold_space
            x.append(x_coordinate)
            y.append(y_coordinate)
            z.append(z_coordinate)
        x=np.array(x)
        y=np.array(y)
        z=np.array(z)
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


    def get_rotation_mtx(self, multi_thread, child_link_quat_pyquat, parent_link_quat_pyquat, child_link_pose_wf):
        if multi_thread=="GPU_simple":
            quat_pyquat = child_link_quat_pyquat * parent_link_quat_pyquat
            rotation_matrix = quat_pyquat.rotation_matrix
            tx, ty, tz = child_link_pose_wf[0].item(), child_link_pose_wf[1].item(), child_link_pose_wf[2].item()
            O_link_frame_wf = np.array([tx, ty, tz])
            T = [tx, ty, tz]
            matrice_homogene = concat_rotation_t_matrix(rotation_matrix=rotation_matrix, T=T)

        else:
            parent_link_quat_pyquat_array = parent_link_quat_pyquat
            child_link_quat_pyquat_array = child_link_quat_pyquat
            child_link_pose_wf_array = child_link_pose_wf
            quat_pyquat_torch_concat = torch.empty(0, 4)
            rotation_matrix_torch_concat =  torch.empty(0, 3,3)
            matrice_homogene_torch_concat = torch.empty(0, 4,4)

            for i in range(len(child_link_quat_pyquat)):
                parent_link_quat_pyquat = parent_link_quat_pyquat_array[i].cpu().numpy()
                child_link_quat_pyquat = child_link_quat_pyquat_array[i].cpu().numpy()
                child_link_pose_wf = child_link_pose_wf_array[i].cpu().numpy()

                quat_pyquat_torch, rotation_matrix_torch, matrice_homogene_torch = quaternion_product(
                    child_link_quat_pyquat=child_link_quat_pyquat,
                    parent_link_quat_pyquat=parent_link_quat_pyquat,
                    child_link_pose_wf=child_link_pose_wf)#array
                print("%%%%%%%%%%",i,"%%%%%%")
                quat_pyquat_torch_concat =  torch.cat((quat_pyquat_torch_concat, quat_pyquat_torch.unsqueeze(0)), dim=0) #torch.stack((quat_pyquat_torch_concat, quat_pyquat_torch ),dim=0)
                rotation_matrix_torch_concat = torch.cat((rotation_matrix_torch_concat, rotation_matrix_torch.unsqueeze(0)), dim=0)  # torch.stack((rotation_matrix_torch_concat, rotation_matrix_torch), dim=0)
                matrice_homogene_torch_concat = torch.cat((matrice_homogene_torch_concat, matrice_homogene_torch.unsqueeze(0)), dim=0)  # torch.stack ((matrice_homogene_torch_concat, matrice_homogene_torch), dim=0)
            matrice_homogene = matrice_homogene_torch_concat
            quat_pyquat = quat_pyquat_torch_concat
        return matrice_homogene, quat_pyquat, O_link_frame_wf



    def apply_rotation_arround_articulation(self,sim_scene, nbr_item_joint_studied=3, multi_thread=None,  geometric_debug=True,direction="positive"):
        display_debug_geometry = True
        display_force_array = False
        trajectory_geometric_debug = True #display trajectroy motion
        debug_opening = True
        if display_debug_geometry:
            nstep=10
        else:
            nstep=1000
        small_force_appliend_on_fingers = -0.05
        hard_force_appliend_on_fingers = -1.5#0-1.5
        sim_scene.command_open_fingers(multi_thread)
        for i in range(nstep):
            print("i = ", i)
            sim_scene.apply_force_on_robot(small_force_appliend_on_fingers, multi_thread)
            sim_scene.command_remain_origin(multi_thread)
            if display_force_array:
                contact_arrows = sim_scene.robot.get_links_net_contact_force()
                contact_left_arrow = contact_arrows[1].cpu().numpy()
                contact_right_arrow = contact_arrows[2].cpu().numpy()
                print("contact_left_arrow", contact_left_arrow)
                print("contact_right_arrow", contact_right_arrow)

                sim_scene.scene.draw_debug_arrow([0, 0, 0], 10 * contact_left_arrow, radius=0.0006, color=vert)
                sim_scene.scene.draw_debug_arrow([0, 0, 0], 10 * contact_right_arrow, radius=0.0006, color=rouge)
            sim_scene.scene.step()
        print("End of applying small control force")
        child_link_quat_pyquat, parent_link_quat_pyquat, child_link_pose_wf,dof_motion_angle_lf = sim_scene.get_parent_child_info(multi_thread,nbr_item_joint_studied)
        matrice_homogene, quat_pyquat,O_link_frame_wf = self.get_rotation_mtx( multi_thread, child_link_quat_pyquat, parent_link_quat_pyquat, child_link_pose_wf)
        pdb.set_trace()
       # torch_matrice_homogene = torch.tile(torch.tensor([100], device=gs.device), (sim_scene.n_envs, 1))
       # torch_object_remain_origin = torch.tile(torch.tensor([0, 0, 0, 0, 0, 0], device=gs.device),(sim_scene.n_envs, 1))

        if geometric_debug:
         sim_scene.scene.draw_debug_frame(matrice_homogene, axis_length=2, origin_size=0.015, axis_radius=0.001)

        B = np.absolute(dof_motion_angle_lf) #si valeur absolue est - ou plsu va changer
        value_active_axis_local_ref = np.argmax(B)
        sign = np.sign(dof_motion_angle_lf[0][value_active_axis_local_ref])
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


        ref_axis_orthogonal_to_active_direction_world_frame = quat_pyquat.rotate(
            ref_axis_orthogonal_to_active_direction_local_frame)
        ref_axis_orthogonal2_to_active_direction_world_frame = quat_pyquat.rotate(
            ref_axis_orthogonal2_to_active_direction_local_frame)
        ref_axis_active_direction_world_frame = quat_pyquat.rotate(ref_axis_active_direction_local_frame)

        quat_array= np.array([quat_pyquat[0],quat_pyquat[1],quat_pyquat[2],quat_pyquat[3]])
        ref_axis_orthogonal_to_active_direction_world_frame2 = quaternion_rotate_vector(
            quat_array,
            ref_axis_orthogonal_to_active_direction_local_frame)
        ref_axis_orthogonal2_to_active_direction_world_frame2 = quaternion_rotate_vector(
            quat_array,
            ref_axis_orthogonal2_to_active_direction_local_frame)
        ref_axis_active_direction_world_frame2 = quaternion_rotate_vector(
            quat_array,
            ref_axis_active_direction_local_frame)


        pdb.set_trace()


        ref_X_LinkFrame_in_wf = quat_pyquat.rotate(ref_axis_end_x)
        ref_Y_LinkFrame_in_wf = quat_pyquat.rotate(ref_axis_end_y)
        ref_Z_LinkFrame_in_wf = quat_pyquat.rotate(ref_axis_end_z)

        if geometric_debug:
            sim_scene.scene.draw_debug_arrow(O_link_frame_wf, 0.3*ref_X_LinkFrame_in_wf, radius=0.01, color=(1, 0, 0, 0.5))
            sim_scene.scene.draw_debug_arrow(O_link_frame_wf, 0.3*ref_Y_LinkFrame_in_wf, radius=0.01, color=(0, 1, 0, 0.5))
            sim_scene.scene.draw_debug_arrow(O_link_frame_wf, 0.3*ref_Z_LinkFrame_in_wf, radius=0.01, color=(0, 0, 1.0, 0.5))

        num_points = 100
        sim_scene.scene.draw_debug_sphere(np.array(O_link_frame_wf), radius=0.1, color=(1.0, 0.0, 0.0, 0.5))
        O_point_wf = O_link_frame_wf
        Mprime_point_wf = sim_scene.robot.get_pos().cpu().numpy()

        R_point_wf = O_point_wf + ref_axis_orthogonal_to_active_direction_world_frame
        R2_point_wf = O_point_wf + ref_axis_orthogonal2_to_active_direction_world_frame

        A_point_wf = O_point_wf + ref_axis_active_direction_world_frame
        OA_vect = A_point_wf - O_point_wf

        OR_vect_wf = R_point_wf - O_point_wf
        OR2_vect_wf = R2_point_wf - O_point_wf
        OMprime_vect = Mprime_point_wf - O_point_wf

        sim_scene.scene.draw_debug_arrow(O_point_wf, OMprime_vect, radius=0.006, color=jaune)

        OMprime_on_OA_vect= np.dot(OMprime_vect,OA_vect)*OA_vect
        OMprime_on_R2_vect= np.dot(OMprime_vect,OR2_vect_wf)*OR2_vect_wf
        OMprime_on_R_vect = np.dot(OMprime_vect,OR_vect_wf)*OR_vect_wf

        Mprime_on_OA_point= O_point_wf + OMprime_on_OA_vect
        H_point = Mprime_on_OA_point
        HMprime_vect  = Mprime_point_wf - H_point
        Mprime_on_OR2_point =  O_point_wf + OMprime_on_R2_vect
        Mprime_on_OR_point = O_point_wf + OMprime_on_R_vect

        if display_debug_geometry :

            sim_scene.scene.draw_debug_sphere(Mprime_on_OA_point, radius=0.01, color=mauve)
            sim_scene.scene.draw_debug_sphere(Mprime_on_OR2_point, radius=0.01, color=mauve)
            sim_scene.scene.draw_debug_sphere(Mprime_on_OR_point, radius=0.01, color=mauve)
            sim_scene.scene.draw_debug_sphere(R2_point_wf, radius=0.01, color=mauve)
            sim_scene.scene.draw_debug_sphere(R_point_wf, radius=0.01, color=mauve)

            sim_scene.scene.draw_debug_sphere(H_point, radius=0.01, color=mauve)
            sim_scene.scene.draw_debug_sphere(Mprime_point_wf, radius=0.01, color=mauve)

            sim_scene.scene.draw_debug_arrow(O_point_wf, OMprime_on_OA_vect, radius=0.006, color=mauve)
            sim_scene.scene.draw_debug_arrow(O_point_wf, OMprime_on_R2_vect, radius=0.006, color=mauve)
            sim_scene.scene.draw_debug_arrow(H_point, HMprime_vect, radius=0.006, color=mauve)
            sim_scene.scene.draw_debug_arrow(O_point_wf, HMprime_vect, radius=0.006, color=mauve)
            sim_scene.scene.draw_debug_arrow(Mprime_on_OR_point, OMprime_on_R2_vect, radius=0.006, color=mauve)

            nom_radius_around_axis = np.linalg.norm(HMprime_vect)
            vect_reference_radius_theta_zero = nom_radius_around_axis * OR2_vect_wf

            theta_rad=np.arccos(np.dot(OR2_vect_wf,HMprime_vect)/(np.linalg.norm(OR2_vect_wf)*np.linalg.norm(HMprime_vect)))
            rot_vect_in_wf = np.cross(OR2_vect_wf,HMprime_vect)
            value_rot_axis_local_ref = np.argmax(np.abs(rot_vect_in_wf))
            sign_rot = np.sign(rot_vect_in_wf[value_rot_axis_local_ref])

            value_active_axis_local_ref = np.argmax(np.abs(OA_vect))
            sign_active_axis = np.sign(OA_vect[value_active_axis_local_ref])
            if sign_active_axis!=sign_rot:
                theta_rad=-theta_rad

            sim_scene.scene.draw_debug_arrow(H_point, 3*HMprime_vect, radius=0.0006, color=bordeaux)
            sim_scene.scene.draw_debug_arrow(O_point_wf, 3*OMprime_on_R2_vect, radius=0.0006, color=bordeaux)
            sim_scene.scene.draw_debug_arrow(H_point, 3 * HMprime_vect, radius=0.03, color=bordeaux)
            sim_scene.scene.draw_debug_arrow(O_point_wf, 3 * OR2_vect_wf, radius=0.03, color=bordeaux)
            #sim_scene.scene.draw_debug_arrow(O_point_wf, 3 * OMprime_on_R2_vect, radius=0.006, color=bordeaux)


            start_angle_deg = theta_rad * 180 / np.pi
            delta_deg_angle = 45

            if direction=="positive":
                stop_angle_deg = start_angle_deg + delta_deg_angle
            elif direction=="negative":
                stop_angle_deg = start_angle_deg - delta_deg_angle
            else:
                raise TypeError('No good direction given')

            stop_angle_rad = stop_angle_deg*np.pi/180

            test_lispace = np.linspace(0,theta_rad,20)
            for theta_rad in test_lispace:
                rotation_matrix_start = self.rotation_matrix_theta_around_axisU(u=OA_vect, theta_rad=theta_rad)
                rotation_matrix_stop = self.rotation_matrix_theta_around_axisU(u=OA_vect, theta_rad=stop_angle_rad)
                start_vect  = np.dot(rotation_matrix_start, vect_reference_radius_theta_zero)
                stop_vect = np.dot(rotation_matrix_stop, vect_reference_radius_theta_zero)

                sim_scene.scene.draw_debug_arrow(O_point_wf, start_vect, radius=0.006, color=vert_canard)
                sim_scene.scene.draw_debug_arrow(O_point_wf, stop_vect, radius=0.006, color=vert_canard)

            radius = nom_radius_around_axis
            center_point = H_point
        else:
            center_point = O_point_wf
            start_angle_deg =10
            stop_angle_deg =20

        x_list, y_list, z_list = self.generate_arc( sim_scene=sim_scene,
                                               ref_vect_for_arc=vect_reference_radius_theta_zero,
                                               u=OA_vect,
                                               stop_angle = stop_angle_deg,
                                               start_angle = start_angle_deg,
                                               num_points = 100,
                                               center_point = H_point)


        if trajectory_geometric_debug:
            if debug_opening:
                num_points = 2
            else:
                pass
            for i in range(num_points):
                x = x_list[i]
                y = y_list[i]
                z = z_list[i]
                if i==0:
                    sim_scene.scene.draw_debug_sphere(np.array([x, y, z]), radius=0.03, color=[0,1,0,1])
                elif i==(num_points-1):
                    sim_scene.scene.draw_debug_sphere(np.array([x, y, z]), radius=0.03, color=[1, 0, 0, 1])
                else:
                    sim_scene.scene.draw_debug_sphere(np.array([x, y, z]), radius=0.01, color=jaune)


        n_waypoint = 0

        sim_scene.robot.set_pos([x_list[n_waypoint], y_list[n_waypoint], z_list[n_waypoint]])
        print("stop debugging, hard force application will start")

        n_step = num_points
        init_action_object_joint_values = sim_scene.object.get_qpos()
        if multi_thread != "GPU_parallel":
            for i in range(n_step):
                pos_robot_in_trajectory = [x_list[n_waypoint], y_list[n_waypoint], z_list[n_waypoint]]
                sim_scene.robot.set_dofs_kp(kp=np.array([100, 100, 100]),dofs_idx_local=[0, 1, 2] )
                sim_scene.robot.control_dofs_position(np.array(pos_robot_in_trajectory), [0, 1, 2])
                for _ in range(100):
                    sim_scene.robot.control_dofs_force(np.array([hard_force_appliend_on_fingers, hard_force_appliend_on_fingers]), sim_scene.finger_items_number)
                    sim_scene.object.set_dofs_position([0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5])
                    sim_scene.scene.step()

                n_waypoint += 1
            difference_init_end_action_joint_value = sim_scene.how_actionable_grasp(
                    multi_thread=multi_thread,
                    init_action_object_joint_values=init_action_object_joint_values)
                #sim_scene.scene.clear_debug_objects()


        else:
            #TODO (A faire plus tard en torch...)
            torch_init_action_object_joint_values = sim_scene.object.get_qpos()
            for i in range(n_step):
                torch_push = torch.tile(torch.tensor([100], device=gs.device), (sim_scene.n_envs, 1))
                torch_object_remain_origin = torch.tile(torch.tensor([0, 0, 0, 0, 0, 0], device=gs.device),
                                                        (sim_scene.n_envs, 1))
                sim_scene.object.set_dofs_position(torch_object_remain_origin, [0, 1, 2, 3, 4, 5])
                sim_scene.robot.control_dofs_force(torch_push, [self.joint_action_indx])
                sim_scene.scene.step()
                revolute_joint_axis = 0

                if i == (n_step - 1):
                    difference_init_end_action_joint_value = sim_scene.how_actionable_grasp(
                        multi_thread=multi_thread,
                        init_action_object_joint_values=torch_init_action_object_joint_values)
        print("last i", i)

        return difference_init_end_action_joint_value


    def rotate_around_axis(self,sim_scene):
        index_object_articulation = 2
        difference_init_end_action_joint_value = self.sim_scene.apply_rotation_arround_articulation(
            nbr_item_joint_studied=index_object_articulation)




    def from_pyquat_to_genesis(self,pyquat):
        genesis_quat = np.array([pyquat[3], pyquat[0], pyquat[1], pyquat[2]])
        return genesis_quat



