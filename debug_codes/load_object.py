import sys
import pdb
from bdb import set_trace

import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np
import time
import os
import pandas as pd
import scipy
import json

from scipy.spatial.transform import Rotation as R
from os import listdir
from os.path import isfile, join

def load_object_position(loader, path_obj_to_grasp, genotype_pose_object,scene):

    object = loader.load(path_obj_to_grasp)
    tabletop_pose = sapien.Pose(
        p=genotype_pose_object[0:3],
        q=genotype_pose_object[3:8]
    )
    object.set_pose(tabletop_pose)
    return object

    # Redimensionner chaque lien


def load_mass(object):
    articulation_object_links = object.get_links()

    # Calcul de la masse totale
    total_mass_object = sum(link.mass for link in articulation_object_links)
    return total_mass_object

def main():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()

    scene_config.gravity = np.array([0.0, 0.0, 0])
    scene = engine.create_scene(scene_config)
    # A small timestep for higher control accuracy
    scene.set_timestep(1 / 2000.0)
    # scene.add_ground(0)

    scene.set_ambient_light([0.5, 0.5, 0.5])

    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)
    object_to_grasp = "7310"


    # Load URDF
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = True

    path_panda = "/home/mathilde/Documents/codes/qd_action_project/robots/panda_gripper.urdf"
    robot = loader.load(path_panda)

    genotype_pose_robot =  [-0.733, -0.309, 0.052, 0.059, 0.720, -0.026, 0.691] # [-0.5, 0, 0, 1, 0, 0, 0]
    tabletop_pose = sapien.Pose(
        p=genotype_pose_robot[0:3],
        q=genotype_pose_robot[3:8]
    )
    robot.set_pose(tabletop_pose)
    nbr_obj =  7310
    lababo ="/home/mathilde/Documents/codes/qd_action_project/partnet-mobility-dataset/" +str(nbr_obj) + "/mobility.urdf"
    rescaledobj =  lababo #"/home/mathildek/Documents/codes/qd_action_project/PartNetMobility_partial_dataset/100658/mobility.urdf"#"/home/mathildek/Documents/codes/articulated_object_sapien/object_sapien/3398_rescaled/mobility.urdf"

    genotype_pose_object = [0,
     0,
     0,
     1,
     0,
     0,
     0]
    object = load_object_position(loader, rescaledobj, genotype_pose_object, scene)
    #TODO VOIR PK savon s actionne pas
    """

    active_joints_robot = robot.get_active_joints()
    init_joint_value = [0] * len(active_joints_robot)
    init_joint_value[2] = 2

    active_joints_object = object.get_active_joints()
    init_joint_value = [0] * len(active_joints_object)
    #active_joints_object[0].set_drive_target(2)
    #active_joints_object[0].set_drive_property(stiffness=2, damping=2, force_limit=100000, mode="force")

    mass_object = load_mass(object)
    mass_robot = load_mass(robot)
    #active_joints_robot[2].set_drive_target(-1)
    #active_joints_robot[2].set_drive_property(stiffness=20000, damping=5000, force_limit=100000, mode="force")
    #active_joints_robot = object.get_active_joints()
    #active_joints_robot[0].set_drive_property(stiffness=1, damping=1, force_limit=1, mode="force")
    """
    for i in range(1000000):
        print(i)
        #active_joints_robot[2].set_drive_velocity_target(+3)
        #active_joints_robot[2].set_drive_target(0)
        scene.step()
        scene.update_render()  # Mettre à jour le rendu (facultatif)
        viewer.render()

if __name__ == '__main__':
    main()