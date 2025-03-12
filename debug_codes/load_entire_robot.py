
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
plane = scene.add_entity(
    gs.morphs.Plane(),
)

franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)
assembly_path ="/home/mathilde/Documents/codes/qd_action_project_2/partnet-mobility-dataset/100658/mobility.urdf"
offset_carton = np.array([0, 1.3, 0.31881])
carton = scene.add_entity(
    gs.morphs.URDF(file=assembly_path, pos=(offset_carton[0], offset_carton[1], offset_carton[2])),
)

scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# set control gains
# Note: the following values are tuned for achieving best behavior with Franka
# Typically, each new robot would have a different set of parameters.
# Sometimes high-quality URDF or XML file would also provide this and will be parsed.
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
)
end_effector = franka.get_link('hand')
consigne = np.array([-0.376, -0.661, 0.402, -0.189, -0.089, 0.824, 0.526])
# move to pre-grasp pose
scene.draw_debug_sphere(consigne[:3]+offset_carton, radius=0.05, color=(0, 0, 1, 0.5))

qpos = franka.inverse_kinematics(
    link = end_effector,
    pos  = consigne[:3]+offset_carton,#np.array([0.65, 0.0, 0.25]),
    quat = consigne[3:] #np.array([0, 1, 0, 0]),
)
fingers_dof = np.arange(7, 9)
# gripper open pos
qpos[-2:] = 0.04
path = franka.plan_path(
    qpos_goal     = qpos,
    num_waypoints = 200, # 2s duration
)
for i in range(5):
    pos = carton.links[i].pos+offset_carton
    scene.draw_debug_sphere(pos, radius=0.05, color=(0, 1, 0.0, 0.5))


# execute the planned path
for waypoint in path:
    franka.control_dofs_position(waypoint)
    scene.step()

for i in range(100):
    print(i)
    scene.step()

tensor_kp = franka.get_dofs_kp()
tensor_kp_sclaed = tensor_kp*(0)
franka.set_dofs_kp(tensor_kp_sclaed.cpu().numpy()[:7],[0,1,2,3,4,5,6])
carton.set_friction(5)
franka.set_friction(5)
for i in range(100000):
    franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
    scene.step()
    print(i)

pdb.set_trace()
