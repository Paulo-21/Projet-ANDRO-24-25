
import sys
import pdb
from bdb import set_trace
import numpy as np
import genesis as gs
import os
import torch



gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0, 0, 0),
    ),
    show_viewer=True,
    show_FPS=False,
)




assembly_path ="/home/mathilde/Documents/codes/onehsape_to_urdf_mathilde/assembly/robot.urdf"
assembly = scene.add_entity(
    gs.morphs.URDF(file=assembly_path, pos=(0, 2, 0)),
)


path_test_obj = "/home/mathilde/Documents/codes/qd_action_project_2/partnet-mobility-dataset/100658_test/mobility.urdf"

test_obj = scene.add_entity(
    gs.morphs.URDF(file=path_test_obj, pos=(0, 0, 0)),

)
path_obj = "/home/mathilde/Documents/codes/qd_action_project_2/partnet-mobility-dataset/100658/mobility.urdf"

obj = scene.add_entity(
    gs.morphs.URDF(file=path_obj, pos=(4, 0, 0)),

)
path_panda ="/home/mathilde/Documents/codes/qd_action_project_2/robots/panda_gripper.urdf"
gripper = scene.add_entity( gs.morphs.URDF(file=path_panda, pos=(3, 0, 0)),)
infos = [(obj.joints.__getitem__(i).name,obj.joints.__getitem__(i).dof_idx_local) for i in range(len(obj.joints))]
nb_iter = 100
range_list = np.linspace(0,1.3,nb_iter)

scene.build()
pdb.set_trace()
for i in range(1000000):
    print(i)
    for i in range(nb_iter):
        obj.control_dofs_position([range_list[i]], [6])
        test_obj.control_dofs_position([range_list[i]], [6])

        #robot.get_dofs_force_range()
        #robot.control_dofs_force([10], [6])
        #robot.get_dofs_force([6])
        #articulated_object.get_dofs_position([range_list[i]], [6])
    # active_joints_robot[2].set_drive_velocity_target(+3)
    # active_joints_robot[2].set_drive_target(0)
        for i in range(10):
            scene.step()
