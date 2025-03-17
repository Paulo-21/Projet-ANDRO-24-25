import pdb

import numpy as np
from .individual import Individual
import multiprocessing
import torch
from ..object_informations.object_informations import *
from ..utils_algo_genetique.population import *
from ..utils_algo_genetique.archive import Archive
from ..utils_algo_genetique.population import Population
from ..utils_bootstrap.bootstrap_related import *
from ..object_informations.object_informations import *
import argparse


from ..utils_geomety_sapien.sapien_scene_simulation import Sapien_scene_simualtion
from ..utils_geometry_genesis.genesis_scene_simulation import Genesis_scene_simulation

class QD_algorithm(Individual):
    def __init__(self,**infos):
        self.__dict__.update(infos)
        self.archive=None
        self.pop=None
        self.simulator_scene_simualtion=None

    def init_QD_algo(self):
        self.archive = Archive(dynamic_application=self.dynamic_application, stock_path=self.stock_path)
        self.pop = Population(size=self.pop_size, artificial_bb=self.artificial_bb)
        if self.simu == "sapien":
            self.simulator_scene_simualtion = Sapien_scene_simualtion(self.gripper, self.object_to_grasp)
        elif self.simu == "genesis":
            self.simulator_scene_simualtion = Genesis_scene_simulation(self.gripper, self.object_to_grasp)
        else :
            raise ValueError("simu mode does not exist")
        self.simulator_scene_simualtion.setup_scene(render_mode=self.render_mode, multi_thread=self.multi_thread)
        self.simulator_scene_simualtion.set_bb(self.artificial_bb)

    def first_generation(self):
        self.pop.initialize_population(bootstrap=self.bootstrap, simulator_scene_simualtion=self.simulator_scene_simualtion)
        self.archive.create_csv_archive_file(
            dynamic_application=self.dynamic_application,
            simulator_scene_simualtion=self.simulator_scene_simualtion,
            simulator=self.simu,
        stock_path=self.stock_path)
        self.pop.evaluate_population(
            simulator_scene_simualtion=self.simulator_scene_simualtion,
            instance_of_archive=self.archive,
            multi_thread=self.multi_thread,
            generation_mode=self.generation_mode,
            simulator=self.simu,
            nbr_item_joint_studied=self.nbr_item_joint_studied,
        )

    def genereation_iteration_process(self):
        for generation in range(self.nb_generations):
            print('GENERATION = ', generation)
            if self.biased_selection:
                selected_individuals_to_be_mutated = self.archive.select_best_individuals_from_archive(
                    size_selection=self.pop.pop_size)
            else:
                selected_individuals_to_be_mutated = self.archive.select_random_individuals_from_archive(
                    size_selection=self.pop.pop_size)
            self.pop.mutate_population_from_selction(selected_individuals_to_be_mutated, self.coefxyz_mutation)
            self.pop.evaluate_population(
                simulator_scene_simualtion=self.simulator_scene_simualtion,
                instance_of_archive=self.archive,
                multi_thread=self.multi_thread,
                generation_mode=self.generation_mode,
                simulator=self.simu,
            nbr_item_joint_studied=self.nbr_item_joint_studied)
        self.archive.store_archive_in_csv(action_mode=self.dynamic_application)
        print('end_generation')



