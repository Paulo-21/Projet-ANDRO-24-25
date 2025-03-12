import pdb
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import csv
from ..trajectory_motion.trajectory_planner import Trajectory_planner


class Individual():
    def __init__(self,size_individual, artificial_bb):
        self.ind_size = size_individual
        self.borne_min_unnormalized_xyz = -artificial_bb
        self.borne_max_unnormalized_xyz= artificial_bb
        self.borne_min_unnormalized_quat = -1
        self.borne_max_unnormalized_quat = 1
        self.artificial_bb = artificial_bb

    def from_normalized_to_unormalized(self, ind_norm_part, mode):
        if mode == "xyz":
            my_a = self.borne_min_unnormalized_xyz
            my_b = self.borne_max_unnormalized_xyz
        elif mode == 'quat':
            my_a = self.borne_min_unnormalized_quat
            my_b = self.borne_max_unnormalized_quat
        unormalized_inidvidual_muattion = my_a * np.ones(np.size(ind_norm_part)) + (my_b-my_a)*ind_norm_part
        return unormalized_inidvidual_muattion

    def affect_fitness_from_difference_init_end_action_joint_value_one_sample_scene_push_action(
            self,
            difference_init_end_action_joint_value,
            individual_genotype,
            action,
            simulator_scene_simulation,
            generation_mode,
            result_queue):
        difference_init_end_action_joint_value = difference_init_end_action_joint_value.tolist()
        norme_1 = np.sum(np.abs(difference_init_end_action_joint_value))
        fitness_info = norme_1
        if norme_1 > 0.01:
            fitness = norme_1  # on peut aussi choisir fitness discrette zero ou 1
            if generation_mode:
                self.store_genotype(individual_genotype=individual_genotype,
                                    fitness_info=fitness_info,
                                    action_mode=action,
                                    simulator_scene_simulation=simulator_scene_simulation)
        else:
            fitness = 0
        feature_decriptor = self.define_behavior_from_genotype(genotype=individual_genotype)
        archive_element = {"genome": individual_genotype, "fitness": fitness,
                           "behavior_descriptor": feature_decriptor}

        if result_queue is not None:
            result_queue.put(archive_element)
        return archive_element

    def evaluate_action_for_one_individual_shaking_action(self,simulator_scene_simualtion,
                                                                                       individual_genotype,
                                                                                       archive,
                                                                                       result_queue,
                                                                                       generation_mode,
                                                                                       multi_thread):

        fitness_info_close = simulator_scene_simualtion.close_finger_action(multi_thread=multi_thread)
        if fitness_info_close!=[True,True]:
            fitness_info_close_and_shake=0
        else :
            fitness_info_close_and_shake = simulator_scene_simualtion.shake_action(multi_thread=multi_thread)

        pdb.set_trace()
        archive_element = self.affect_fitness_from_left_right_touching_finger_one_sample_scene_close_finger(
            fitness_info, individual_genotype, simulator_scene_simualtion, result_queue, generation_mode,
            action_mode=archive.dynamic_application)
        return archive_element

    def evaluate_action_for_one_individual_rotate_around_joint(self, action, multi_thread, individual_genotype, simulator_scene_simulation, generation_mode, result_queue, nbr_item_joint_studied):
        geometric_debug=True
        direction = "positive" #"negative"
        my_trajectory_planner = Trajectory_planner()
        difference_init_end_action_joint_value = my_trajectory_planner.apply_rotation_arround_articulation(nbr_item_joint_studied=nbr_item_joint_studied,
                                                                                     multi_thread=multi_thread,
                                                                                     sim_scene=simulator_scene_simulation,
                                                                                     geometric_debug=geometric_debug,
                                                                                     direction=direction)

        archive_element = self.affect_fitness_from_difference_init_end_action_joint_value_one_sample_scene_push_action(
            difference_init_end_action_joint_value=difference_init_end_action_joint_value,
            individual_genotype=individual_genotype,
            action=action,
            simulator_scene_simulation=simulator_scene_simulation,
            generation_mode=generation_mode,
            result_queue=result_queue

        )
        return archive_element

    def evaluate_action_for_one_individual_push_action(self, simulator_scene_simulation,individual_genotype,archive, result_queue, generation_mode, action, multi_thread):

        difference_init_end_action_joint_value = simulator_scene_simulation.push_action(action=action,multi_thread=multi_thread)
        archive_element = self.affect_fitness_from_difference_init_end_action_joint_value_one_sample_scene_push_action(
            difference_init_end_action_joint_value=difference_init_end_action_joint_value,
            individual_genotype =individual_genotype,
            action=action,
            simulator_scene_simulation=simulator_scene_simulation,
            generation_mode=generation_mode,
            result_queue=result_queue

        )
        return archive_element

    def affect_fitness_from_left_right_touching_finger_one_sample_scene_close_finger(self,fitness_info, individual_genotype, simulator_scene_simualtion,  result_queue, generation_mode, action_mode):
        [rigth_finger_is_touching, left_finger_is_touching] = fitness_info
        if (rigth_finger_is_touching or left_finger_is_touching):
            fitness = 1
            if generation_mode:
                self.store_genotype(individual_genotype, fitness_info, action_mode=action_mode,
                                    simulator_scene_simulation=simulator_scene_simualtion)
        else:
            fitness = 0
        feature_decriptor = self.define_behavior_from_genotype(genotype=individual_genotype)
        archive_element = {"genome": individual_genotype, "fitness": fitness,
                           "behavior_descriptor": feature_decriptor}

        if result_queue is not None:
            result_queue.put(archive_element)
            return result_queue
        else:
            return archive_element

    def evaluate_action_for_one_individual_close_finger_action(self, simulator_scene_simualtion, multi_thread, individual_genotype=None,archive=None, result_queue=None, generation_mode=None):
        fitness_info = simulator_scene_simualtion.close_finger_action(multi_thread=multi_thread)
        archive_element = self.affect_fitness_from_left_right_touching_finger_one_sample_scene_close_finger(fitness_info, individual_genotype, simulator_scene_simualtion, result_queue, generation_mode, action_mode=archive.dynamic_application)
        return archive_element


    def evaluate_action_for_one_individual(self, simulator_scene_simualtion, individual_genotype, result_queue, archive, generation_mode,simulator, multi_thread, nbr_item_joint_studied):
        simulator_scene_simualtion.load_object(multi_thread=multi_thread)
        simulator_scene_simualtion.load_robot(individual_genotype,multi_thread=multi_thread)
        simulator_scene_simualtion.define_joint_and_scalar_to_activate_close_action()

        collisions = simulator_scene_simualtion.is_there_colision_object_end_effector()
        if collisions:
            archive_element = {"genome": individual_genotype, "fitness": 0,
                               "behavior_descriptor":  self.define_behavior_from_genotype(genotype=individual_genotype)}
        else :
            if archive.dynamic_application == "close_finger":
                archive_element = self.evaluate_action_for_one_individual_close_finger_action(simulator_scene_simualtion=simulator_scene_simualtion,
                                                                                       individual_genotype=individual_genotype,
                                                                                       archive=archive,
                                                                                       result_queue=result_queue,
                                                                                       generation_mode=generation_mode,
                                                                                       multi_thread=multi_thread)

            elif archive.dynamic_application == "push_right" or archive.dynamic_application == "push_left" or archive.dynamic_application == "push_forward" or archive.dynamic_application=="push_backward" :
                archive_element = self.evaluate_action_for_one_individual_push_action(simulator_scene_simulation=simulator_scene_simualtion,
                                                            individual_genotype=individual_genotype,
                                                            archive=archive,
                                                            result_queue=result_queue,
                                                            generation_mode=generation_mode,
                                                           action= archive.dynamic_application,
                                                            multi_thread=multi_thread)

            elif archive.dynamic_application == "rotate_around_joint":
                archive_element = self.evaluate_action_for_one_individual_rotate_around_joint( action=archive.dynamic_application,
                                                                                               multi_thread=multi_thread,
                                                                                               individual_genotype=individual_genotype,
                                                                                               simulator_scene_simulation=simulator_scene_simualtion,
                                                                                               generation_mode=generation_mode,
                                                                                               result_queue=result_queue,
                                                                                               nbr_item_joint_studied=nbr_item_joint_studied)
            elif  archive.dynamic_application == "shaking":
                archive_element = self.evaluate_action_for_one_individual_shaking_action(
                    simulator_scene_simualtion=simulator_scene_simualtion,
                    individual_genotype=individual_genotype,
                    archive=archive,
                    result_queue=result_queue,
                    generation_mode=generation_mode,
                    multi_thread=multi_thread)


            else :
                raise ValueError('the dynamic mode you choose doesn t exist')
            if generation_mode==False:
                if archive_element["fitness"] == 1:
                    color = [0, 1, 0]
                else:
                    color = [1, 0, 0]
                simulator_scene_simualtion.add_debug_sphere_success_falure_genotype(color, individual_genotype)

        return archive_element

    def store_genotype(self, individual_genotype,fitness_info,action_mode, simulator_scene_simulation):
        path_csv = simulator_scene_simulation.csv_scene_name
        new_line = pd.DataFrame([[individual_genotype, fitness_info, action_mode]], index=None)
        with open(path_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(new_line.iloc[0].tolist())

    def define_behavior_from_genotype(self,genotype):
        feature = genotype[0:3]
        feature_rounded = np.round(feature, 3)
        return feature_rounded

    def from_normalized_to_scaled_mutation(self, rand_nomalized_mutation):
        mutation_scaled = rand_nomalized_mutation * self.artificial_bb
        return mutation_scaled
