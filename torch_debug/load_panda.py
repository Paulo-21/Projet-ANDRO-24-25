import pdb

import numpy as np
import genesis as gs
import os
import torch
########################## init ##########################

"""
my_object = scene.add_entity(
    gs.morphs.URDF(file=os.getcwd() + '/partnet-mobility-dataset/' + '10143' + '/mobility.urdf',
                   pos=(0, 0, 0.8057),
                   # euler=(0, 0, 0),  # we follow scipy's extrinsic x-y-z rotation convention, in degrees,
                   quat=(1.0, 0.0, 0.0, 0.0),  # we use w-x-y-z convention for quaternions,
                   scale=1.0,
                   ), )
"""
import numpy as np


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)  #qo : w

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

def concat_rotation_t_matrix(rotation_matrix, T):
    tx, ty, tz = T[0], T[1], T[2]
    matrice_homogene = np.array([[rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], tx],
                                 [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], ty],
                                 [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], tz],
                                 [0, 0, 0, 1]],
                                )
    return matrice_homogene
def main():
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options = gs.options.ViewerOptions(
            camera_pos    = (3, -1, 1.5),
            camera_lookat = (0.0, 0.0, 0.5),
            camera_fov    = 30,
            max_FPS       = 60,
        ),
        sim_options = gs.options.SimOptions(
            dt = 0.01,
        ),
        show_viewer = True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    bootstrap_pos_quat = [-0.534, 0.234, 0.205+0.8057, 0.430, 0.536, 0.455, 0.567]
    T = bootstrap_pos_quat[:3]
    bootstrap_quat = bootstrap_pos_quat[3:]

    rot_mtx = quaternion_rotation_matrix(bootstrap_quat)
    TF = concat_rotation_t_matrix(rot_mtx,T)

    franka = scene.add_entity(
        gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml',  pos= (-1,0,0.5)),

    )
    ########################## build ##########################
    scene.build()
    scene.draw_debug_frame(TF)
    pdb.set_trace()
    debug_grasp = np.array([-0.534, 0.234, 0.205+0.8057])
    scene.draw_debug_sphere(debug_grasp, radius=0.1, color=(1.0, 0.0, 0.0, 0.5))

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

    # get the end-effector link
    end_effector = franka.get_link('hand')

    # move to pre-grasp pose
    qpos = franka.inverse_kinematics(link = end_effector, pos  = np.array(bootstrap_pos_quat),quat = np.array([0, 1, 0, 0]),)
    # gripper open pos
    qpos[-2:] = 0.04
    path = franka.plan_path(
        qpos_goal     = qpos,
        num_waypoints = 200, # 2s duration
    )
    pdb.set_trace()
    jnt_names = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
        'finger_joint1',
        'finger_joint2',
    ]
    dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]
    franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
    # execute the planned path
    for waypoint in path:
        franka.control_dofs_position(waypoint)
        scene.step()

    # allow robot to reach the last waypoint
    for i in range(100000):
        scene.step()


if __name__ == '__main__':
    main()