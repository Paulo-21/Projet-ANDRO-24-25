import pdb
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import csv
import json

import os
import sapien.core as sapien
from sapien.utils.viewer import Viewer
from ..actions_achitecture_module.actions_architectures import *
from ..utils_bootstrap.bootstrap_related import *



class Sapien_scene_simualtion():
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
        self.finger_items_tuple = None
        self.finger_items_number = None
        self.active_joints_robot =None
        self.active_joints_object = None

    def create_csv_scene_file(self, dynamic_application):
        base_path = os.getcwd() + "/results/"
        csv_scene_name =base_path + "object_" + self.object_to_grasp + "_robot_" +  self.gripper  + "_"+ dynamic_application + ".csv"

        version = 1
        while os.path.exists(csv_scene_name):
            version += 1
            new_file_name = f"object_{self.object_to_grasp}_robot_{self.gripper}_{dynamic_application}_v{version}.csv"
            csv_scene_name = os.path.join(base_path, new_file_name)

        self. csv_scene_name = csv_scene_name
        file = open(self.csv_scene_name, 'w+')

        return csv_scene_name


    def init_object_path(self):
        return os.getcwd() + "/PartNetMobility_partial_dataset/" + self.object_to_grasp + '/mobility.urdf'

    def init_robot_path(self, gripper):
        curent_working_directory = os.getcwd()
        if gripper=="panda":
            self.path_panda = curent_working_directory + "/robots/panda_gripper.urdf"
        else :
            raise ValueError(f"Gripper '{gripper}' non pris en charge. Veuillez utiliser 'panda'.")
        return self.path_panda

    def setup_scene(self,render_mode, multi_thread):
        engine = sapien.Engine()
        if render_mode==True:
            renderer = sapien.SapienRenderer()
            engine.set_renderer(renderer)
        else:
            renderer=None

        scene_config = sapien.SceneConfig()
        print(scene_config.gravity)

        scene_config.gravity = np.array([0.0, 0.0, 0])
        print('it works')
        self.scene = engine.create_scene(scene_config)
        # A small timestep for higher control accuracy
        self.scene.set_timestep(1 / 2000.0)

        if render_mode:
            self.scene.set_ambient_light([0.5, 0.5, 0.5])
            self.scene.set_ambient_light([0.5, 0.5, 0.5])
            self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
            self.viewer = Viewer(renderer)
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(x=-2, y=0, z=1)
            self.viewer.set_camera_rpy(r=0, p=-0.3, y=0)
        else :
            self.viewer=None

        self.loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        self.loader.fix_root_link = True

        return self.scene, self.loader, self.viewer

    def set_bb(self, artificial_bb):
        path_object_to_grasp = os.getcwd() + '/PartNetMobility_partial_dataset/' + self.object_to_grasp
        path_object_to_grasp_bbox = path_object_to_grasp + "/bounding_box.json"

        with open(path_object_to_grasp_bbox) as f:
            dict_bbox = json.load(f)
            print(dict_bbox)
        lenght = (dict_bbox["max"][0] - dict_bbox["min"][0])
        height = (dict_bbox["max"][1] - dict_bbox["min"][1])
        width = (dict_bbox["max"][2] - dict_bbox["min"][2])
        w = width / 2
        l = lenght / 2
        h = height / 2
        list_bb = [w, h, l]  # [w,l,h] #TODO:verifier les bb (bounding box)
        archive_meter_side = max(lenght, height, width)
        if self.viewer is not None:
            self.viewer.draw_aabb(np.array([-artificial_bb, -artificial_bb, -artificial_bb]), np.array([artificial_bb, artificial_bb, artificial_bb]), np.array([0, 0, 0]))
        else:
            pass
        return lenght, height, width, archive_meter_side

    def load_object(self, multi_thread=None, object_st="test_obj_str"):
        int_obj_pose = [0, 0, 0, 1, 0, 0, 0]
        tabletop_pose = sapien.Pose(
            p=int_obj_pose[0:3],
            q=int_obj_pose[3:8])
        if self.object is None:
            self.object = self.loader.load(self.path_object_to_grasp)
            self.object.set_name("my_object")
            self.object.set_pose(tabletop_pose)
        else :
            self.object.set_pose(tabletop_pose)
            previous_qvel = [0] * len(self.active_joints_object)
            self.object.set_qvel(previous_qvel)
            self.object.set_qpos(previous_qvel)
            self.object.set_qf(previous_qvel)

    def load_robot(self,individual_genotype, multi_thread=None):
        tabletop_pose = sapien.Pose(
            p=individual_genotype[0:3],
            q=individual_genotype[3:8])
        if self.robot is None:
            self.robot = self.loader.load(self.robot_path)
            self.robot.set_name("my_robot")
            self.robot.set_pose(tabletop_pose)
        else:
            self.robot.set_pose(tabletop_pose)
            previous_qvel =  [0] * len(self.active_joints_robot)
            self.robot.set_qvel(previous_qvel)
            self.robot.set_qf(previous_qvel)

    def close_command_without_execution(self):
        self.define_joint_and_scalar_to_activate_close_action()

        init_joint_value = [0] * len(self.active_joints_robot)
        init_joint_value[-1], init_joint_value[-2] = 0.035, 0.035
        self.robot.set_qpos(init_joint_value)

        for indx in self.finger_items_number:
            self.active_joints_robot[indx].set_drive_property(stiffness=100, damping=5, force_limit=100000,
                                                              mode="force")
            self.active_joints_robot[indx].set_drive_property(stiffness=100, damping=5, force_limit=100000,
                                                              mode="force")
            self.active_joints_robot[indx].set_drive_target(0)

    def close_finger(self):

        self.close_command_without_execution()

        rigth_finger_is_touching = False
        left_finger_is_touching = False

        for i in range(100):
            self.scene.step()
            self.scene.update_render()
            if self.viewer is not None:
                self.viewer.render()
            if (i == 99):
                [rigth_finger_is_touching, left_finger_is_touching] = self.find_fingers_contact()
            self.scene.step()
            self.scene.update_render()



        return [rigth_finger_is_touching, left_finger_is_touching]

    def define_joint_and_scalar_to_activate_close_action(self):
        self.active_joints_robot =  self.robot.get_active_joints()
        self.active_joints_object = self.object.get_active_joints()
        self.finger_items_tuple = [(i, self.active_joints_robot.__getitem__(i).get_name()) for i in range(len( self.active_joints_robot)) if ( self.active_joints_robot.__getitem__(i).get_name()=='panda_finger_joint1' or  self.active_joints_robot.__getitem__(i).get_name()=='panda_finger_joint2')]
        self.finger_items_number = [i for (i, j) in self.finger_items_tuple]



    def push_action(self, action, multi_thread):
        "close and push in reality"
        self.close_command_without_execution()
        max_step = 300
        for i in range(max_step):
            self.scene.step()
            self.scene.update_render()
            if self.viewer is not None:
                self.viewer.render()
            self.scene.step()

        #la commande de femeture continue pendant le push, TODO changer stifness de femeture pour qu elle soit plus ferme, maispas entrer en concurence avec push

        scalar_push_value, joint_action_indx = self.sapien_action_architecture(action)
        self.active_joints_robot[joint_action_indx].set_drive_property(stiffness=200, damping=500, force_limit=100000, mode="force")
        self.active_joints_robot[joint_action_indx].set_drive_target(scalar_push_value)
        init_action_object_joint_values = self.get_object_joint_values_()

        for i in range(max_step):
            self.scene.step()
            self.scene.update_render()
            if self.viewer is not None:
                self.viewer.render()

            if (i == max_step-1):
                difference_init_end_action_joint_value= self.how_actionable_grasp( init_action_object_joint_values)
            else:pass

            self.scene.step()
            self.scene.update_render()

        return difference_init_end_action_joint_value

    def create_sphere(self,
            scene: sapien.Scene,
            pose: sapien.Pose,
            radius,
            color=None,
            name="",
    ) -> sapien.Entity:
        """Create a sphere. See create_box."""
        builder = scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, material=color)
        sphere = builder.build(name=name)
        sphere.set_pose(pose)
        return sphere

    def analyse_contact(self):
        contacts = self.scene.get_contacts()
        if contacts:
            for i in range(len(contacts)):
                point_position = contacts.__getitem__(i).points.__getitem__(0).position

                entity_in_contact = [body.get_articulation().get_name() for body in contacts.__getitem__(i).bodies]
                if (entity_in_contact[0] == "my_robot" and entity_in_contact[1] == 'my_object') or (
                        entity_in_contact[0] == 'my_object' and entity_in_contact[1] == "my_robot"):
                    print("object is touching the end effector, because of collison, the individual will not be tested")
                    collision = True
                else :
                    collision = False
                return collision
    def is_there_colision_object_end_effector(self, ):
        for i in range(1):
            self.scene.step()
            self.scene.update_render()
        collision = self.analyse_contact()
        return collision

    def get_object_joint_values_(self):
        values_joints = self.object.get_qpos()
        return  values_joints

    def how_actionable_grasp(self, init_action_object_joint_values):
        '''
        Difference entre les etats des articulations au debut et a la fin
        '''

        end_action_object_joint_values = self.get_object_joint_values_()
        diffence_init_end_action_joint_value = init_action_object_joint_values - end_action_object_joint_values

        return diffence_init_end_action_joint_value

    def find_fingers_contact(self):
        contacts = self.scene.get_contacts()
        rigth_finger_is_touching = False
        left_finger_is_touching = False
        for contact in contacts:
            if ((contact.bodies[0].entity.name == "panda_rightfinger") and (
                    contact.bodies[1].entity.name != "panda_leftfinger")) or (
                    (contact.bodies[1].entity.name == "panda_rightfinger") and (
                    contact.bodies[0].entity.name != "panda_leftfinger")):
                rigth_finger_is_touching = True

            if ((contact.bodies[0].entity.name == "panda_leftfinger") and (
                    contact.bodies[1].entity.name != "panda_rightfinger")) or (
                    (contact.bodies[1].entity.name == "panda_leftfinger") and (
                    contact.bodies[0].entity.name != "panda_rightfinger")):
                left_finger_is_touching = True

        return [rigth_finger_is_touching, left_finger_is_touching]


    def remove_robot(self):
        pdb.set_trace()
        self.scene.remove_articulation(self.robot)
    def remove_object(self):
        self.scene.remove_articulation(self.object)

    def add_debug_sphere_success_falure_genotype(self, color, individual_genotype):
        sphere = self.create_sphere(
            self.scene,
            sapien.Pose(p=individual_genotype[0:3]),  # sapien.Pose(p=[0, -0.2, 1.0 + 0.05]),
            radius=0.01,
            color=color,
            name="sphere",
        )

    def remove_robot_scene(self):
        self.remove_robot()

    def remove_object_scene(self):
        self.remove_object()

    def sapien_action_architecture(self,action):
        '''genesis : [([0, 1, 2, 3, 4, 5], 'joint_base_link'), (6, 'translationX'), (7, 'translationY'), (8, 'translationZ'), (9, 'rotationX'), (10, 'rotationY'), (11, 'rotationZ'), (12, 'panda_finger_joint1'), (13, 'panda_finger_joint2')]'''
        scale_push_motion = 4
        if action == "push_left":
            scalar_push_value = 1 * scale_push_motion
            joint_action_indx = 0
        elif action == "push_right":
            scalar_push_value = -1 * scale_push_motion
            joint_action_indx = 0
        elif action == "push_forward":
            scalar_push_value = 1 * scale_push_motion
            joint_action_indx = 2
        elif action == "push_backward":
            scalar_push_value = -1 * scale_push_motion
            joint_action_indx = 2
        else:
            raise Exception("Sorry, push mode is invalid")
        return scalar_push_value, joint_action_indx
