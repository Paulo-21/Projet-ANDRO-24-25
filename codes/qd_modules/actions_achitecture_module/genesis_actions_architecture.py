import pdb

import numpy as np


def genesis_action_achitecture(simulation_scene_simulator, action):
    scale_push_motion = 4
    if action == "push_forward":
        joint_action_indx = simulation_scene_simulator.translationZ_items_number
        scalar_push_value = 1 * scale_push_motion
    elif action == "push_backward":
        joint_action_indx = simulation_scene_simulator.translationZ_items_number
        scalar_push_value = -1 * scale_push_motion
    elif action == "push_up":
        joint_action_indx = simulation_scene_simulator.translationY_items_number
        scalar_push_value = 1 * scale_push_motion
    elif action =="push_down":
        joint_action_indx = simulation_scene_simulator.translationY_items_number
        scalar_push_value = -1 * scale_push_motion
    elif action == "push_left":
        joint_action_indx = simulation_scene_simulator.translationX_items_number
        scalar_push_value = 1 * scale_push_motion
    elif action == "push_right":
        joint_action_indx = simulation_scene_simulator.translationX_items_number
        scalar_push_value = -1 * scale_push_motion
    elif action == "push_right":
        joint_action_indx = simulation_scene_simulator.translationX_items_number
        scalar_push_value = -1 * scale_push_motion
    return scalar_push_value,joint_action_indx
