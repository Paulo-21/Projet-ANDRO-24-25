import pdb

import numpy as np
import time
import os
import pandas as pd
os.environ["PYOPENGL_PLATFORM"] = "glx"

import glob


from qd_modules.utils_algo_genetique.population import *
from qd_modules.utils_algo_genetique.archive import Archive
from qd_modules.utils_algo_genetique.population import Population
from qd_modules.utils_algo_genetique.QD_algorithm import QD_algorithm
from qd_modules.utils_bootstrap.bootstrap_related import *
from qd_modules.object_informations.object_informations import *
from qd_modules.utils_archive_analysis.archive_analysis import ArchiveAnalysis
import argparse
from PIL import Image
simu = "genesis"

if simu == "sapien":
    from qd_modules.utils_geomety_sapien.sapien_scene_simulation import Sapien_scene_simualtion
if simu == "genesis":
    from qd_modules.utils_geometry_genesis.genesis_scene_simulation import Genesis_scene_simulation


"""
    else :
        if version is not None:
            path_csv_success = os.getcwd() + '/result_archive/' + simu + '_object_' + object_to_grasp + '_robot_' + gripper +'_' + dynamic_application + "_v" + str(version) +'.csv'
        else :
            path_csv_success = os.getcwd() + '/result_archive/' + simu + '_object_' + object_to_grasp + '_robot_' + gripper +'_' + dynamic_application +'.csv'
        bootstraped_individuals = read_successful_grasp(path_csv_success)
        pop.restart_pop_from_bootraped_individuals(bootstraped_individuals)
        pop.replay_individuals(simulator_scene_simualtion,archive,simulator=simu,  indx=replay_indx, multi_thread=multi_thread, action=dynamic_application)
        pop.display_map_3d(sapien_scene_simualtion)
"""

def argument_management():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--multi_thread', default='GPU_parallel', choices=['GPU_parallel', 'GPU_simple', 'all'], help='')
    parser.add_argument('--render_mode', default='False', choices=['True', 'False'], help='')
    parser.add_argument('--generation_mode', default='False', choices=['True', 'False'], help='')
    parser.add_argument('--version', default='None', choices=['3', '4'], help='')
    parser.add_argument('--replay_indx', default='None', choices=['3', '4'], help='')
    parser.add_argument('--obj_name', default='carton', choices=['grille_pain', 'laptop', 'carton', 'Refrigerator'], help='')
    parser.add_argument('--action', default='close_finger',
                        choices=['close_finger', 'rotate_around_joint', 'push_forward', 'push_right','shaking'], help='')
    parser.add_argument('--robot', default='end_effector', choices=['entire_robot','end_effector'])
    parser.add_argument('--viz', default='True', choices=['True','False'])
    parser.add_argument('--stock_path', default='data/analysis_qd_action')
    parser.add_argument('--mutation_operator', default='gaussian', choices=['gaussian', 'random','es','sa','cov'], help='Mutation method')


    args = parser.parse_args()
    print(vars(args))
    viz = args.viz
    stock_path = args.stock_path
    multi_thread = args.multi_thread
    render_mode = args.render_mode
    generation_mode = args.generation_mode
    version = args.version
    replay_indx = args.replay_indx
    obj_name = args.obj_name
    action = args.action
    robot = args.robot

    mutation_operator = args.mutation_operator

    if multi_thread == True and render_mode:
        raise ValueError(
            "Erreur : multi_thread est activé mais render_mode est désactivé. Veuillez activer render_mode pour utiliser multi_thread.")
    print("before QD")
    return multi_thread, render_mode, generation_mode, version, replay_indx, obj_name, action, robot, viz, stock_path, mutation_operator

def main():
    artificial_bb = 2 * 0.5 # TODO Faire une bounding box de la taille

    multi_thread, render_mode, generation_mode, version, replay_indx, obj_name, action, robot, viz, stock_path, mutation_operator= argument_management()


    my_QD_algo = QD_algorithm(Name="Fist_name",
            biased_selection = True,
            nb_generations=1000,
            pop_size=100,
            coefxyz_mutation = 0.01,
            scale_mutation=0.05,
            mutation_operator=mutation_operator,
            dynamic_application= action,
            bootstrap=True,
            object_to_grasp= DICT_OBJECT[obj_name],
            multi_thread=multi_thread,
            render_mode=render_mode,
            generation_mode=generation_mode,
            gripper=robot,
            artificial_bb = artificial_bb,
            replay_indx=replay_indx,
            version = version,
            simu=simu,
            nbr_item_joint_studied=1,
                              stock_path=stock_path
            )
    my_QD_algo.init_QD_algo()
    if viz=="False":
        my_QD_algo.first_generation()
        my_QD_algo.genereation_iteration_process()

    elif viz=="True" :
        
            path_root = my_QD_algo.stock_path + "/genesis_object_100658_robot_end_effector_close_finger_cov_unknown.csv"
            my_analysis = ArchiveAnalysis(path_root)
            #my_analysis = ArchiveAnalysis(csv_file)
            my_QD_algo.simulator_scene_simualtion.init_robot_path(my_QD_algo.gripper)
            my_QD_algo.simulator_scene_simualtion.load_object(multi_thread=multi_thread)
            my_QD_algo.simulator_scene_simualtion.load_robot(np.array([2, 0, 0, 1, 0, 0, 0]), multi_thread=multi_thread)
            to_plot_genomes_True= my_analysis.data_frame[my_analysis.data_frame.fitness == 1]["genome"]
            to_plot_genomes_False = my_analysis.data_frame[my_analysis.data_frame.fitness == 0]["genome"]
            print(len(to_plot_genomes_True))
            if my_QD_algo.gripper =="end_effector":
                offset= np.array([0, 0, 0])
            elif my_QD_algo.gripper =="entire_robot":
                offset = np.array([0, 1.3, 0.31881])
            else:
                raise ValueError("pb gripper")

            for item in range(len(to_plot_genomes_True)):
                str_genome = to_plot_genomes_True.iloc[item]
                genome_sphere = np.fromstring(str_genome.strip("[]").replace("\n", " "), sep=" ")
                my_QD_algo.simulator_scene_simualtion.scene.draw_debug_sphere(genome_sphere[0:3]+offset, radius=0.005, color=(0, 1.0, 0.0, 0.5))
            print(len(to_plot_genomes_False))
            rgb, depth, segmentation, normal = my_QD_algo.simulator_scene_simualtion.cam.render()
            image = Image.fromarray(rgb)
            image.save(my_QD_algo.stock_path+ "_green.png")
            #image.save(csv_file.replace(".csv", "_green.png"))

            for item in range(len(to_plot_genomes_False)):
                str_genome = to_plot_genomes_False.iloc[item]
                genome_sphere = np.fromstring(str_genome.strip("[]").replace("\n", " "), sep=" ")
                my_QD_algo.simulator_scene_simualtion.scene.draw_debug_sphere(
                    genome_sphere[0:3] +offset, radius=0.005, color=(1, 0.0, 0.0, 0.5))
                print("logg", item, "/",len(to_plot_genomes_False) )
            rgb, depth, segmentation, normal = my_QD_algo.simulator_scene_simualtion.cam.render()
            image_end=Image.fromarray(rgb)
            #image_end.save(csv_file.replace(".csv", "_green_red.png"))
            image_end.save(my_QD_algo.stock_path +"_green_red.png")

            for i in range(1000):
                my_QD_algo.simulator_scene_simualtion.cam.render()
                my_QD_algo.simulator_scene_simualtion.scene.step()
    else :
        raise ValueError("viz value")

    # pdb.set_trace()


if __name__ == '__main__':
    main()