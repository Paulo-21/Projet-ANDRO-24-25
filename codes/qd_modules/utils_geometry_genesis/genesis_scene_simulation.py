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
from ..trajectory_motion import *

class Genesis_scene_simulation():
    def __init__(self, gripper, object_to_grasp):
        self.gripper = gripper
        self.object_to_grasp = object_to_grasp
        self.scene = None
        self.robot_path = self.init_robot_path(gripper)
        self.viewer = None
        self.path_object_to_grasp = self.init_object_path()
        self.robot = None
        self.object = None
        self.csv_scene_name = None #self.csv_scene_name()
        self.bootstrap_dictionary = BOOTSTRAP_DICTIONARY_HANDMADE
        self.gs_device = None
        self.finger_items_tuple = None
        self.finger_items_number =None
        self.n_envs = None
        self.translationX_items_number=None
        self.translationY_items_number = None
        self.translationZ_items_number = None
        self.all_tuple_object = None
        self.all_tuple_robot = None

    def init_robot_path(self, gripper):
        curent_working_directory = os.getcwd()
        if gripper=="panda":
            self.path_panda = curent_working_directory + "/robots/panda_gripper.urdf"
        else :
            raise ValueError(f"Gripper '{gripper}' non pris en charge. Veuillez utiliser 'panda'.")
        return self.path_panda

    def init_object_path(self):
        return os.getcwd() + "/PartNetMobility_partial_dataset/" + self.object_to_grasp + '/mobility.urdf'

    def create_csv_scene_file(self, dynamic_application):
        base_path = os.getcwd() + "/results/"
        csv_scene_name =base_path + "genesis_object_" + self.object_to_grasp + "_robot_" +  self.gripper  + "_"+ dynamic_application + ".csv"

        version = 1
        while os.path.exists(csv_scene_name):
            version += 1
            new_file_name = f"genesis_object_{self.object_to_grasp}_robot_{self.gripper}_{dynamic_application}_v{version}.csv"
            csv_scene_name = os.path.join(base_path, new_file_name)

        self. csv_scene_name = csv_scene_name
        file = open(self.csv_scene_name, 'w+')

        return csv_scene_name

    def load_object(self,multi_thread):
        if self.object is None :
            self.object = self.scene.add_entity(
                gs.morphs.URDF(file=os.getcwd() + '/PartNetMobility_partial_dataset/' + self.object_to_grasp + '/mobility.urdf',
                               pos=(0, 0, 0),
                              # euler=(0, 0, 0),  # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
                               quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
                               scale=1.0,
                               ), )
        else :
            #self.scene.reset()
            if multi_thread != "GPU_parallel":
                self.object.set_pos(np.array([0, 0, 0]))
                self.object.set_quat( np.array([1.0, 0.0, 0.0, 0.0]))
            else :
                torch_pos= torch.tile(torch.tensor([0, 0,0], device=gs.device), (self.n_envs, 1))
                torch_quat = torch.tile(torch.tensor([1, 0, 0, 0], device=gs.device), (self.n_envs, 1))
                self.object.set_pos(torch_pos)
                self.object.set_quat(torch_quat)
                print("######################### INIT OBJECT #########################")


    def load_robot(self,individual_genotype, multi_thread):
        pos = tuple(individual_genotype[0:3])
        quat = tuple(individual_genotype[3:])
       # q2 = Quaternion(axis=[0, 0, 1], angle=3.14159265 / 2)
        #quat = quat * q2
        if self.robot is None:
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(file= os.getcwd() +'/robots/panda_gripper.urdf',
                               pos=pos,
                               #euler=(0, 0, 90),  # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
                                quat  =quat, # we use w-x-y-z convention for quaternions,
                               scale=1.0,
                               ), )

            print("##### begin build")
            self.scene.build()
            print("##### end build")
        else:
            self.robot.set_pos(pos)
            self.robot.set_quat(quat)


    def load_robot_parallel(self,pop_size):
        if self.robot is None:
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(file=os.getcwd() + '/robots/panda_gripper.urdf',
                               pos=(3,3,3),
                               # euler=(0, 0, 90),  # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
                               quat=(1,0,0,0),  # we use w-x-y-z convention for quaternions,
                               scale=1.0,
                               ), )
            print("##### begin build parallel")
            self.n_envs = pop_size
            self.scene.build(n_envs=self.n_envs, env_spacing=(5.0, 5.0))
            print("##### end build parallel")
        else :
            torch_pos = torch.tile(torch.tensor([0, 0, 0], device=gs.device), (self.n_envs, 1))
            torch_quat = torch.tile(torch.tensor([1, 0, 0, 0], device=gs.device), (self.n_envs, 1))
            self.robot.set_pos(torch_pos)
            self.robot.set_quat(torch_quat)

    def set_bb(self, artificial_bb):
        "TODO"
        return 0

    def setup_scene(self, multi_thread, render_mode):
        if multi_thread=="GPU_parallel" or  multi_thread=="GPU_simple":
         gs.init(backend=gs.gpu)
         self.gs_device = gs.gpu
        elif multi_thread=="CPU_simple" or multi_thread=="CPU_multi_thread" :
            gs.init(backend=gs.cpu)
        else :
            raise Exception("please enter a good value for multi_thread")
        self.scene = gs.Scene(
            show_FPS=False,
            sim_options=gs.options.SimOptions(
                dt=0.01,
                gravity = (0,0,0),
            ),
            show_viewer=render_mode,
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,

            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=True,  # visualize the coordinate frame of `world` at its origin
                world_frame_size=1.0,  # length of the world frame in meter
                show_link_frame=True,  # do not visualize coordinate frames of entity links
                show_cameras=True,  # do not visualize mesh and frustum of the cameras added
                plane_reflection=True,  # turn on plane reflection
                ambient_light=(0.1, 0.1, 0.1),  # ambient light setting
            ),

        )


    def is_there_colision_object_end_effector(self):\
        return False #TODO


    def define_joint_and_scalar_to_activate_close_action(self):
        joints_list_robot = self.robot.joints
        test_debug_dimension =[(self.robot.joints.__getitem__(i).dof_idx_local, self.robot.joints.__getitem__(i).name) for i in range(len(joints_list_robot)) ]

        self.all_tuple_robot = [(self.robot.joints.__getitem__(i).dof_idx_local, self.robot.joints.__getitem__(i).name) for i in
                              range(len(joints_list_robot))]
        self.all_tuple_object = [(self.object.joints.__getitem__(i).dof_idx_local, self.object.joints.__getitem__(i).name)
                                for i in
                                range(len(self.object.joints))]
        self.finger_items_tuple = [(self.robot.joints.__getitem__(i).dof_idx_local, self.robot.joints.__getitem__(i).name) for i in
                              range(len(joints_list_robot)) if
                              self.robot.joints.__getitem__(i).name == "panda_finger_joint1" or self.robot.joints.__getitem__(
                                  i).name == "panda_finger_joint2"]
        self.finger_items_number = [i for (i, j) in self.finger_items_tuple] #12 et 13 :)
        self.translationX_items_number = 0
        self.translationY_items_number = 1
        self.translationZ_items_number = 2
        """
        self.translationX_items_number = [self.robot.joints.__getitem__(i).dof_idx_local for i in
                                                                    range(len(joints_list_robot)) if
                                                                    self.robot.joints.__getitem__(
                                                                        i).name == "translationX"][0]
        self.translationY_items_number = [self.robot.joints.__getitem__(i).dof_idx_local
                                                                    for i in
                                                                    range(len(joints_list_robot)) if
                                                                    self.robot.joints.__getitem__(
                                                                        i).name == "translationY"][0]
        self.translationZ_items_number = [self.robot.joints.__getitem__(i).dof_idx_local
                                                                    for i in
                                                                    range(len(joints_list_robot)) if
                                                                    self.robot.joints.__getitem__(
                                                                        i).name == "translationZ"][0]
        """
        self.set_scene_physics_properties()
        # Question : est ce qu on attache la base fixe pour pas que l objet se fasse emporter ou est ce qu on met plus de poid su l objet a la base plutot (pour l instant éeme cas)
        #TODO same for rotation :)

    def set_scene_physics_properties(self):
        self.object.set_friction(1.0)
        self.robot.set_friction(1.0)
        masses_object = self.object.get_links_inertial_mass()
        masses_object_scaled = masses_object
        masses_object_scaled[0] = masses_object[0] * 10000
        self.object.set_links_inertial_mass(masses_object_scaled)

        self.object.set_friction(1.0)
        masses_robot = self.robot.get_links_inertial_mass()
        masses_robot_scaled = masses_robot

        self.robot.set_links_inertial_mass(masses_robot_scaled)

    def close_finger_action(self,multi_thread):
        "mode parallel and not parallel"
        if multi_thread!="GPU_parallel":
            self.robot.set_dofs_position(np.array([0.035, 0.035]), self.finger_items_number)
            self.robot.set_dofs_kp(
                kp=np.array([1000, 1000]),
                dofs_idx_local=self.finger_items_number,
            )
            self.robot.set_dofs_kv(
                kv=np.array([5, 5]),
                dofs_idx_local=self.finger_items_number,
            )

            for i in range(120):
                self.robot.control_dofs_force(
                    np.array([-1, -1]),
                    self.finger_items_number,
                )
                self.robot.set_dofs_position(np.array([0]), [0])
                self.scene.step()
                if (i == 99):
                    [rigth_finger_is_touching, left_finger_is_touching] = self.find_fingers_contact(multi_thread=multi_thread)
            return [rigth_finger_is_touching, left_finger_is_touching]

        else :
            #torch_friction = torch.tile(torch.tensor([1], device=gs.device), (self.n_envs, 1))
            #self.object.set_friction(torch_friction)
            #self.robot.set_friction(1.0)
            torch_open_gripper = torch.tile(torch.tensor([0.035, 0.035], device=gs.device), (self.n_envs, 1))
            self.robot.set_dofs_position(torch_open_gripper, self.finger_items_number)
            for i in range(120):
                torch_force_control = torch.tile(torch.tensor([-1, -1], device=gs.device), (self.n_envs, 1))
                self.robot.control_dofs_force(torch_force_control,self.finger_items_number )

                torch_maintain_pos = torch.tile(torch.tensor([0], device=gs.device), (self.n_envs, 1))
                self.robot.set_dofs_position(torch_maintain_pos, [0])
                self.scene.step()
                if i==119:
                    tensor_rigth_finger_is_touching_left_finger_is_touching = self.find_fingers_contact(multi_thread=multi_thread)

            return tensor_rigth_finger_is_touching_left_finger_is_touching

    def how_actionable_grasp(self,init_action_object_joint_values, multi_thread):
        '''init '''

        end_action_object_joint_values = self.object.get_qpos()
        diffence_init_end_action_joint_value = init_action_object_joint_values - end_action_object_joint_values
        test_debug = [self.robot.joints.__getitem__(i).name for i in range(len(self.robot.joints))]
        return diffence_init_end_action_joint_value #revoie tenseur ou liste selon le cas

    def push_action(self,action, multi_thread):
        #self.close_finger_action(multi_thread)
        scalar_push_value, joint_action_indx = genesis_action_achitecture(simulation_scene_simulator=self,
                                                                          action=action)
        n_step = 300
        if multi_thread !="GPU_parallel":
            init_action_object_joint_values = self.object.get_qpos()
            for i in range(n_step):
                self.robot.control_dofs_force([1],[joint_action_indx])
                array_object_remain_origin =np.array([0,0,0])
                self.object.set_dofs_position(array_object_remain_origin, [0,1,2])
                if (i == n_step-1):
                    difference_init_end_action_joint_value =  self.how_actionable_grasp(multi_thread=multi_thread,
                        init_action_object_joint_values=init_action_object_joint_values)
                self.scene.step()
        else :
            torch_init_action_object_joint_values = self.object.get_qpos()
            for i in range(n_step):
                torch_push = torch.tile(torch.tensor([1], device=gs.device), (self.n_envs, 1))
                torch_object_remain_origin = torch.tile(torch.tensor([0,0,0], device=gs.device), (self.n_envs, 1))
                self.object.set_dofs_position(torch_object_remain_origin, [0,1,2])
                self.robot.control_dofs_force(torch_push, [joint_action_indx])
                self.scene.step()
                if i == (n_step-1):
                    difference_init_end_action_joint_value = self.how_actionable_grasp(
                        multi_thread=multi_thread,
                        init_action_object_joint_values=torch_init_action_object_joint_values)
        return difference_init_end_action_joint_value


    def check_non_null(self,tensor_list):
        list_contact = []
        is_there_contact = False
        for idx, tensor in enumerate(tensor_list):
            if torch.any(tensor != 0):
                list_contact.append(idx)
                is_there_contact = True
        return is_there_contact, list_contact

    def is_left_right_contact_one_sample(self,tensor):
        is_there_contact, indx_contact = self.check_non_null(tensor)
        right_finger_is_touching = False
        left_finger_is_touching = False

        if is_there_contact == False:
            return right_finger_is_touching, left_finger_is_touching
        else:
            if (7 in indx_contact):
                right_finger_is_touching = True
            if (8 in indx_contact):
                left_finger_is_touching = True
            return right_finger_is_touching, left_finger_is_touching

    def find_fingers_contact(self, multi_thread):
        tensor = self.robot.get_links_net_contact_force()
        if multi_thread=="GPU_simple" or multi_thread=="CPU_simple" or multi_thread=="CPU_multi_thread":
            right_finger_is_touching, left_finger_is_touching = self.is_left_right_contact_one_sample(tensor)
            return  right_finger_is_touching, left_finger_is_touching
        elif multi_thread=="GPU_parallel":
            tensor_false = torch.zeros((self.n_envs,2), dtype=torch.bool)
            tensor_right_finger_is_touching_left_finger_is_touching  = self.tensor_left_finger_right_finger_from_conctact_tensor(tensor=tensor,finger_indx="right", tensor_false=tensor_false)
            tensor_right_finger_is_touching_left_finger_is_touching = self.tensor_left_finger_right_finger_from_conctact_tensor(
                tensor=tensor, finger_indx="left", tensor_false=tensor_right_finger_is_touching_left_finger_is_touching)
            return tensor_right_finger_is_touching_left_finger_is_touching

    def tensor_left_finger_right_finger_from_conctact_tensor(self, tensor,finger_indx, tensor_false ):

        if finger_indx=="right":
            finger_indx = -1
            indx_in_lift_right_couple = 0

        else:
            finger_indx = -2
            indx_in_lift_right_couple = 1

        tensor_contact_finger = tensor[:, finger_indx, :]

        tensor_one_norm_by_scene_sample_total = tensor_contact_finger.norm(dim=1)
        number_of_item = torch.nonzero(tensor_one_norm_by_scene_sample_total != 0, as_tuple=False).numel()

        if number_of_item == 1:
            item_scene_with_contact = [torch.nonzero(tensor_one_norm_by_scene_sample_total != 0, as_tuple=False).squeeze().tolist()]
        else:
            item_scene_with_contact = torch.nonzero(tensor_one_norm_by_scene_sample_total != 0,as_tuple=False).squeeze().tolist()

        for item_scene in item_scene_with_contact:
            tensor_false[item_scene][indx_in_lift_right_couple] = True
        return tensor_false

    def remove_robot(self):
        self.scene.entities.remove(self.robot)
        print('remove_robot')

    def remove_object(self):
        print('remove_obj')
        self.scene.entities.remove(self.object)

    def set_up_init_robot_pos_and_quat(self, pop_size, pop_list):
        if isinstance(pop_list, list):
            pop_list =np.array(pop_list)
        else:pass
        pos_list = pop_list[:, [0,1,2]]
        quat_list = pop_list[:, [-4,-3,-2,-1]]
        tensor_pop_pos_elements = torch.tensor(pos_list, dtype=torch.float64)
        tensor_pop_quat_elements = torch.tensor(quat_list, dtype=torch.float64)
        self.robot.set_pos(tensor_pop_pos_elements)
        self.robot.set_quat(tensor_pop_quat_elements)








