import genesis as gs
import pdb
import numpy as np
gs.init(backend=gs.gpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0,0,0),
    ),
    show_viewer=True,
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
)

"""
object = scene.add_entity(
    gs.morphs.URDF(file='/home/mathildek/Documents/codes/qd_action_project/PartNetMobility_partial_dataset/100658/mobility.urdf',
        pos   = (0, 0, 0),
        euler = (0, 0, 90), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
        # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
        scale = 1.0,
    ),)
"""

robot = scene.add_entity(
    gs.morphs.URDF(file='/home/mathildek/Documents/codes/qd_action_project/robots/panda_gripper.urdf',
        pos   = (0, 0, 0),
        euler = (0, 0, 90), # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
        # quat  = (1.0, 0.0, 0.0, 0.0), # we use w-x-y-z convention for quaternions,
        scale = 1.0,
    ),)

#joints_list = object.joints
joints_list_robot = robot.joints
print('before building scene')
finger_items_tuple = [(robot.joints.__getitem__(i).dof_idx_local,robot.joints.__getitem__(i).name) for i in range(len(joints_list_robot)) if robot.joints.__getitem__(i).name=="panda_finger_joint1" or robot.joints.__getitem__(i).name=="panda_finger_joint2"]
finger_items_number =[i for (i,j) in finger_items_tuple]
scene.build()
robot.set_dofs_position(np.array([0.035,0.035]), finger_items_number)
robot.set_dofs_kp(
    kp             = np.array([1000, 1000]),
    dofs_idx_local = finger_items_number,
)
robot.set_dofs_kv(
    kv             = np.array([5, 5]),
    dofs_idx_local = finger_items_number,
)
pdb.set_trace()

for i in range(1000):
    scene.step()