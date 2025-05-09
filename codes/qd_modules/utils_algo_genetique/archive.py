import pdb
import random
import numpy as np
import pandas as pd
import csv
import os

class Archive():
    def __init__(self, dynamic_application, stock_path,mutation_operator,coefxyz_mutation,scale_mutation):
        self.archive_map = dict()
        self.dynamic_application=dynamic_application
        self.orderer_archive = None
        self.stock_path=stock_path

        self.mutation_operator = mutation_operator
        self.coefxyz_mutation = coefxyz_mutation
        self.scale_mutation = scale_mutation

    def store_one_new_element_in_archive(self, new_archive_element):
        key_element = tuple(new_archive_element["behavior_descriptor"])

        if key_element in self.archive_map:
            archive_element_fitness = self.archive_map[key_element]["fitness"]
            new_archive_element_fitness = new_archive_element["fitness"]
            if new_archive_element_fitness > archive_element_fitness:
                self.archive_map[key_element] = new_archive_element
            else:
                pass
        else:
            self.archive_map[key_element] = new_archive_element
        return self.archive_map

    def order_archive_by_fitness(self):
        self.orderer_archive = sorted(
            self.archive_map.values(),  # Récupère les sous-dictionnaires
            key=lambda x: x.get('fitness', float('-inf')),  # Trie par 'fitness' (ou -inf si absent)
            reverse=True  # Ordre décroissant
        )

    def select_best_individuals_from_archive(self, size_selection):
        self.order_archive_by_fitness()
        archive_element_selected_for_mutation =  self.orderer_archive[0:size_selection]
        selected_individuals_to_be_mutated = [archive_element_selected_for_mutation[i]["genome"] for i in
                                 range(len(archive_element_selected_for_mutation))]
        return selected_individuals_to_be_mutated

    def select_random_individuals_from_archive(self, size_selection):
        #size_archive = len(self.archive_map)
        #archive_element_selected_for_mutation = [random.choice(list(self.archive_map.keys())) for _ in range(size_selection) ]
        #selected_individuals_to_be_mutated = [self.archive_map[key]["genome"] for key in
        #                                      archive_element_selected_for_mutation]
        keys = list(self.archive_map)
        selected_individuals_to_be_mutated = [ self.archive_map[random.choice(keys)]["genome"] for _ in range(size_selection) ]
        return selected_individuals_to_be_mutated

    def create_csv_archive_file(self, dynamic_application, simulator_scene_simualtion, simulator, stock_path):
        '''
        base_path = stock_path
        file_name = f"{simulator}_object_{simulator_scene_simualtion.object_to_grasp}_robot_{simulator_scene_simualtion.gripper}_{dynamic_application}_{self.scale_mutation}.csv"
        csv_archive_name = os.path.join(base_path, file_name)
        #csv_archive_name = base_path + simulator+ "_object_" + simulator_scene_simualtion.object_to_grasp + "_robot_" + simulator_scene_simualtion.gripper + "_" + dynamic_application + ".csv"
        version = 1
        while os.path.exists(csv_archive_name):
            version += 1
            new_file_name = f"{simulator}_object_{simulator_scene_simualtion.object_to_grasp}_robot_{simulator_scene_simualtion.gripper}_{dynamic_application}_v{version}.csv"
            csv_archive_name = os.path.join(base_path, new_file_name)

        self.csv_archive_name = csv_archive_name
        simulator_scene_simualtion.csv_scene_name = self.csv_archive_name # add by ziming

        file = open(self.csv_archive_name, 'w+')
        
        '''
        base_path = stock_path

        if self.mutation_operator == "gaussian":
            mutation_value = self.scale_mutation
        elif self.mutation_operator == "random":
            mutation_value = self.coefxyz_mutation
        
            '''
            elif self.mutation_operator == "es":

            elif self.mutation_operator == "sa":

            elif self.mutation_operator == "cov":
            '''
            
        else:
            mutation_value = "unknown"

        file_name = (
            f"{simulator}_object_{simulator_scene_simualtion.object_to_grasp}"
            f"_robot_{simulator_scene_simualtion.gripper}"
            f"_{dynamic_application}_{self.mutation_operator}_{mutation_value}.csv"
        )

        csv_archive_name = os.path.join(base_path, file_name)

        version = 1
        while os.path.exists(csv_archive_name):
            version += 1
            new_file_name = (
                f"{simulator}_object_{simulator_scene_simualtion.object_to_grasp}"
                f"_robot_{simulator_scene_simualtion.gripper}"
                f"_{dynamic_application}_{self.mutation_operator}_{mutation_value}_v{version}.csv"
            )
            csv_archive_name = os.path.join(base_path, new_file_name)

        self.csv_archive_name = csv_archive_name
        simulator_scene_simualtion.csv_scene_name = self.csv_archive_name
        file = open(self.csv_archive_name, 'w+')



    def store_archive_in_csv(self,action_mode):
        save_data = pd.DataFrame.from_dict(self.archive_map)
        save_data2 = save_data.T
        path_root = self.stock_path
        save_data2.to_csv(self.csv_archive_name, index=False)
        print("archive stored")

    def store_several_element_in_archive(self, archive_element_from_one_tensor):
        for archive_element in range(len(archive_element_from_one_tensor)):
            single_archive_element = archive_element_from_one_tensor[archive_element]
            self.store_one_new_element_in_archive(single_archive_element)