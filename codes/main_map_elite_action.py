import pdb

import numpy as np
import time
import os
import pandas as pd

from qd_modules.utils_algo_genetique.population import *
from qd_modules.utils_algo_genetique.archive import Archive
from qd_modules.utils_algo_genetique.population import Population
from qd_modules.utils_bootstrap.bootstrap_related import *
from qd_modules.object_informations.object_informations import *
import argparse
simu = "genesis"

if simu == "sapien":
    from qd_modules.utils_geomety_sapien.sapien_scene_simulation import Sapien_scene_simualtion
if simu == "genesis":
    from qd_modules.utils_geometry_genesis.genesis_scene_simulation import Genesis_scene_simulation


def QD_algo(biased_selection,
            nb_generations,
            object_to_grasp,
            multi_thread,
            render_mode,
            pop_size,
            generation_mode,
            gripper,
            artificial_bb,
            dynamic_application,
            coefxyz_mutation,
            bootstrap,
            simu,
            replay_indx=None,
            version=None
           ):

    archive = Archive(dynamic_application=dynamic_application)
    pop = Population(size=pop_size, artificial_bb=artificial_bb)

    if simu =="sapien":
        sapien_scene_simualtion = Sapien_scene_simualtion(gripper, object_to_grasp)
        simulator_scene_simualtion = sapien_scene_simualtion
    else:
        genesis_scene_simualtion = Genesis_scene_simulation(gripper, object_to_grasp)
        simulator_scene_simualtion = genesis_scene_simualtion

    simulator_scene_simualtion.setup_scene(render_mode=render_mode, multi_thread=multi_thread)
    simulator_scene_simualtion.set_bb(artificial_bb)

    if generation_mode :
        pop.initialize_population(bootstrap=bootstrap, simulator_scene_simualtion=simulator_scene_simualtion)
        simulator_scene_simualtion.create_csv_scene_file(dynamic_application=dynamic_application)
        archive.create_csv_archive_file(
            dynamic_application=dynamic_application,
            simulator_scene_simualtion=simulator_scene_simualtion,
        simulator=simu)
        pop.evaluate_population(
            simulator_scene_simualtion=simulator_scene_simualtion,
            instance_of_archive=archive,
            multi_thread=multi_thread,
            generation_mode=generation_mode,
            simulator=simu,
        )

        ########## All other generations#########

        for generation in range(nb_generations):
            print('GENERATION = ',generation)
            if biased_selection :
                selected_individuals_to_be_mutated = archive.select_best_individuals_from_archive(size_selection = pop.pop_size)
            else:
                selected_individuals_to_be_mutated = archive.select_random_individuals_from_archive(
                    size_selection=pop.pop_size)
            pop.mutate_population_from_selction(selected_individuals_to_be_mutated, coefxyz_mutation)
            pop.evaluate_population(
                simulator_scene_simualtion=simulator_scene_simualtion,
                instance_of_archive=archive,
                multi_thread=multi_thread,
                generation_mode=generation_mode,
                simulator=simu)
        archive.store_archive_in_csv(action_mode=dynamic_application)
        print('end_generation')

    ###############################################################
    ###### READ QD archive already existing #######################
    ###############################################################

    else :
        if version is not None:
            path_csv_success = os.getcwd() + '/result_archive/' + simu + '_object_' + object_to_grasp + '_robot_' + gripper +'_' + dynamic_application + "_v" + str(version) +'.csv'
        else :
            path_csv_success = os.getcwd() + '/result_archive/' + simu + '_object_' + object_to_grasp + '_robot_' + gripper +'_' + dynamic_application +'.csv'
        bootstraped_individuals = read_successful_grasp(path_csv_success)
        pop.restart_pop_from_bootraped_individuals(bootstraped_individuals)
        pop.replay_individuals(simulator_scene_simualtion,archive,simulator=simu,  indx=replay_indx, multi_thread=multi_thread, action=dynamic_application)
        pop.display_map_3d(sapien_scene_simualtion)

def argument_management():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--multi_thread', default='GPU_simple', choices=['GPU_simple', 'GPU_simple', 'all'], help='')
    parser.add_argument('--render_mode', default='True', choices=[True, False], help='')
    parser.add_argument('--generation_mode', default='True', choices=[True, False], help='')
    parser.add_argument('--version', default='None', choices=['3', '4'], help='')
    parser.add_argument('--replay_indx', default='None', choices=['3', '4'], help='')
    parser.add_argument('--obj_name', default='Refrigerator', choices=['grille_pain', 'laptop', 'carton', 'Refrigerator'], help='')
    parser.add_argument('--action', default='rotate_around_joint',
                        choices=['close_finger', 'rotate_around_joint', 'push_forward', 'push_right'], help='')

    args = parser.parse_args()
    print(vars(args))
    multi_thread = args.multi_thread
    render_mode = args.render_mode
    generation_mode = args.generation_mode
    version = args.version
    replay_indx = args.replay_indx
    obj_name = args.obj_name
    action = args.action

    if multi_thread == True and render_mode:
        raise ValueError(
            "Erreur : multi_thread est activé mais render_mode est désactivé. Veuillez activer render_mode pour utiliser multi_thread.")
    print("before QD")
    return multi_thread, render_mode, generation_mode, version, replay_indx, obj_name, action

def main():

    artificial_bb = 2 * 0.5 # TODO Faire une bounding box de la taille

    multi_thread, render_mode, generation_mode, version, replay_indx, obj_name, action= argument_management()

    QD_algo(biased_selection = False,
            nb_generations=3000,
            pop_size=3021,
            coefxyz_mutation=1,
            dynamic_application= action,
            bootstrap=True,
            object_to_grasp= DICT_OBJECT[obj_name],
            multi_thread=multi_thread,
            render_mode=render_mode,
            generation_mode=generation_mode,
            gripper="panda",
            artificial_bb = artificial_bb,
            replay_indx=replay_indx,
            version = version,
            simu=simu
            )


if __name__ == '__main__':
    main()