import pdb

import numpy as np
from .individual import Individual
import multiprocessing
import torch
from ..trajectory_motion.trajectory_planner import Trajectory_planner



class Population(Individual):
    def __init__(self, size, artificial_bb,mutation_operator,coefxyz_mutation,scale_mutation,nb_generations):
        super().__init__(size_individual=7, artificial_bb=artificial_bb)
        self.pop_size=size
        self.pop_list = None
        
        self.mutation_operator = mutation_operator
        self.coefxyz_mutation = coefxyz_mutation
        self.scale_mutation = scale_mutation
        self.nb_generations = nb_generations
        self.coverage = 0.0
        self.current_evaluation = 0
        

    def from_normalized_to_unnormalize_population(self, nomalized_population):
        pop_unormalized = []
        for ind_norm in nomalized_population:
            ind_norm_xyz = ind_norm[:3]
            ind_norm_quat = ind_norm[3:]
            unormalized_inidvidual_xyz = self.from_normalized_to_unormalized(ind_norm_xyz, mode="xyz")
            unormalized_inidvidual_quat = self.from_normalized_to_unormalized(ind_norm_quat, mode="quat")
            unormalized_inidvidual = np.concatenate(
                (unormalized_inidvidual_xyz, unormalized_inidvidual_quat))
            pop_unormalized.append(unormalized_inidvidual)
        return pop_unormalized

    def initialize_population(self, bootstrap, simulator_scene_simualtion):
        if not bootstrap:
            print("NOT BOOTSTRAP")
            rand_nomalized = np.random.rand(self.pop_size, self.ind_size) #entre zero et un
            pop_unormalized = self.from_normalized_to_unnormalize_population(rand_nomalized)
            self.pop_list =pop_unormalized
        else :
            key = simulator_scene_simualtion.object_to_grasp
            if key in simulator_scene_simualtion.bootstrap_dictionary:
                pose  = simulator_scene_simualtion.bootstrap_dictionary[key]
            else :
                print("value boostrap not found, boostrap on   0 0 0 1 0 0 0")
                pose =  np.array([[0, - 0, 0, 1, 0, 0, 0]])
            rand_nomalized = np.random.rand(self.pop_size - 1, self.ind_size)  # entre zero et un
            pop_unormalized = self.from_normalized_to_unnormalize_population(rand_nomalized)
            self.pop_list = np.concatenate((pose, pop_unormalized))
        return self.pop_list

    def evaluate_population(self, simulator_scene_simualtion, instance_of_archive, multi_thread, generation_mode, simulator, nbr_item_joint_studied):
        if multi_thread=="CPU_simple" or multi_thread=="GPU_simple":
            i=0
            for individual_genotype in self.pop_list:
                print("individual nbr :", i)
                archive_element = self.evaluate_action_for_one_individual(simulator_scene_simualtion=simulator_scene_simualtion, individual_genotype=individual_genotype,result_queue=None, generation_mode=generation_mode, archive=instance_of_archive, simulator=simulator, multi_thread=multi_thread,nbr_item_joint_studied=nbr_item_joint_studied)
                instance_of_archive.store_one_new_element_in_archive(archive_element)
                i+=1
        elif multi_thread=="CPU_multi_thread" :
            result_queue = multiprocessing.Queue()
            processes = []
            for individual_genotype in self.pop_list:
                process = multiprocessing.Process(target= self.evaluate_action_for_one_individual, args=( self, simulator_scene_simualtion, individual_genotype, result_queue, instance_of_archive, generation_mode,simulator, multi_thread))
                processes.append(process)
                process.start()
            for process in processes:
                process.join()
            while not result_queue.empty():
                archive_element = result_queue.get()
                instance_of_archive.store_one_new_element_in_archive(archive_element)

        elif multi_thread=="GPU_parallel":
            archive_element_from_one_tensor = self.evaluate_action_for_tensor_population(action=instance_of_archive.dynamic_application, simulator_scene_simualtion=simulator_scene_simualtion)
            instance_of_archive.store_several_element_in_archive(archive_element_from_one_tensor)

    def evaluate_action_for_tensor_population(self, action, simulator_scene_simualtion):
        simulator_scene_simualtion.load_object(multi_thread="GPU_parallel")
        simulator_scene_simualtion.load_robot_parallel(pop_size=self.pop_size)

        simulator_scene_simualtion.torch_set_up_init_robot_pos_and_quat(pop_size=self.pop_size,pop_list= self.pop_list)
        simulator_scene_simualtion.define_joint_and_scalar_to_activate_close_action()
        simulator_scene_simualtion.scene.step()
        if action=="close_finger":
            tensor_fitness_info = simulator_scene_simualtion.close_finger_action(multi_thread="GPU_parallel")
            archive_element_from_one_tensor = self.affect_fitness_from_left_right_touching_finger_tensor_scene_close_finger(fitness_info=tensor_fitness_info)

        elif action == "push_right" or  action == "push_left" or action == "push_forward" :
            tensor_fitness_info = simulator_scene_simualtion.push_action( action=action,multi_thread="GPU_parallel")
            archive_element_from_one_tensor = self.affect_fitness_from_difference_int_end_action_joint_tensor_scene_push_action(
                fitness_info=tensor_fitness_info)
        elif action== "rotate_around_joint":
            geometric_debug = True
            direction = "positive"  # "negative"
            nbr_item_joint_studied=1
            my_trajectory_planner = Trajectory_planner()
            tensor_fitness_info = my_trajectory_planner.apply_rotation_arround_articulation(
                nbr_item_joint_studied=nbr_item_joint_studied,
                multi_thread="GPU_parallel",
                sim_scene=simulator_scene_simualtion,
                geometric_debug=geometric_debug,
                direction=direction)


            self.affect_fitness_from_difference_int_end_action_joint_tensor_scene_push_action(
                fitness_info=tensor_fitness_info)
        else :
            raise ValueError("pas bonne action")

        return archive_element_from_one_tensor


    def mutate_population_from_selction(self, selection, coverage=None):
        coef_classic_mutattion =0.01
        pop_mutated = []

        # ES operator
        sigma_min = 0.025
        sigma0 = 0.5
        tau = 1 / np.sqrt(self.ind_size)

        # SA operator
        start_sa = 0.2
        stop_sa = 0.025
        N_total = self.pop_size * self.nb_generations

        # Cov operator
        start_cov = 0.4 
        stop_cov = 0.025

        for ind in selection:
            '''
            mutation_on_gene_xyz = np.random.rand(1, 3)*coefxyz_mutation
            mutation_on_classic_gene_xyz = np.random.rand(1, self.ind_size-3) * coef_classic_mutattion # entre zero et un
            mutation_on_gene_xyz = np.random.normal(loc=0.0, scale =scale_mutation, size=(1, 3)) # scale **2 = variance
            mutation_on_classic_gene_xyz = np.random.normal(loc=0.0, scale = scale_mutation, size=(1, self.ind_size - 3))
            '''
            if self.mutation_operator == "gaussian":
                mutation_on_gene_xyz = np.random.normal(loc=0.0, scale=self.scale_mutation, size=(1, 3))# scale **2 = variance
                mutation_on_classic_gene_xyz = np.random.normal(loc=0.0, scale=self.scale_mutation, size=(1, self.ind_size - 3))
                rand_nomalized_mutation = np.concatenate([mutation_on_gene_xyz,mutation_on_classic_gene_xyz], axis=1)

            elif self.mutation_operator == "random":
                mutation_on_gene_xyz = np.random.rand(1, 3)*self.coefxyz_mutation
                mutation_on_classic_gene_xyz = np.random.rand(1, self.ind_size-3) * coef_classic_mutattion # entre zero et un
                rand_nomalized_mutation = np.concatenate([mutation_on_gene_xyz,mutation_on_classic_gene_xyz], axis=1)
            
            elif self.mutation_operator == "es":
                sigma = sigma0 * np.exp(tau * np.random.normal(0, 1))
                sigma = max(sigma, sigma_min)
                rand_nomalized_mutation = np.random.normal(loc=0.0, scale=sigma, size=(1, self.ind_size))
     
            elif self.mutation_operator == "sa":
                progress = self.current_evaluation / N_total
                sigma = start_sa - (start_sa - stop_sa) * progress
                sigma = max(sigma, stop_sa)
                rand_nomalized_mutation = np.random.normal(loc=0.0, scale=sigma, size=(1, self.ind_size))
                self.current_evaluation += 1
                
            elif self.mutation_operator == "cov":
                sigma = start_cov - (start_cov - stop_cov) * (coverage if coverage is not None else 0.0)
                sigma = max(sigma, stop_cov)
                rand_nomalized_mutation = np.random.normal(loc=0.0, scale=sigma, size=(1, self.ind_size))


            else:
                raise ValueError(f"Unknown mutation operator: {self.mutation_operator}")

            # rand_nomalized_mutation = np.concatenate([mutation_on_gene_xyz,mutation_on_classic_gene_xyz], axis=1)
            # self.from_normalized_to_scaled_mutation(rand_nomalized_mutation)
            mutation_scaled = self.from_normalized_to_scaled_mutation(rand_nomalized_mutation)
            ind_mutated = ind + mutation_scaled[0]
            pop_mutated.append(ind_mutated)
        self.pop_list = pop_mutated
        return self.pop_list

    def restart_pop_from_bootraped_individuals(self, bootstraped_individuals):
        self.pop_list = bootstraped_individuals

    def replay_individuals(self, simulator_scene_simualtion, archive, simulator, multi_thread, action,indx=None):

        if indx==None:
            i=0
            for individual_genotype in self.pop_list:
                print("individual nbr :", i, "/", len(self.pop_list))
                simulator_scene_simualtion.load_object(object_st=object_st)
                simulator_scene_simualtion.load_robot(individual_genotype, multi_thread=multi_thread)
                simulator_scene_simualtion.define_joint_and_scalar_to_activate_close_action()
                self.evaluate_action_for_one_individual(simulator_scene_simualtion=simulator_scene_simualtion,
                                         individual_genotype=individual_genotype,
                                         result_queue=None,
                                         generation_mode=False,
                                         archive=archive,
                                         simulator=simulator,
                                                        multi_thread=multi_thread)

                i += 1
        else :
            individual_genotype = self.pop_list[indx]
            for i in range(5):
                self.evaluate_action_for_one_individual(simulator_scene_simualtion=simulator_scene_simualtion,
                                         individual_genotype=individual_genotype,
                                         result_queue=None,
                                         generation_mode=False,
                                         archive=archive,
                                         simulator=simulator)

    def display_map_3d(self,sapien_scene_simualtion):
        i=0
        int_obj_pose = [0,0,0,1,0,0,0]
        sapien_scene_simualtion.load_object(int_obj_pose)
        for individual_genotype in self.pop_list:
            print("individual pourcent", i/len(self.pop_list)*100, " TOTAL ", i, " / ",len(self.pop_list) )
            sapien_scene_simualtion.add_debug_sphere_success_falure_genotype([0,1,0], individual_genotype)
            i+=1
        print("end of the program, visualisation still running")
        while not sapien_scene_simualtion.viewer.closed:
            sapien_scene_simualtion.scene.step()
            sapien_scene_simualtion.scene.update_render()
            sapien_scene_simualtion.viewer.render()

    def affect_fitness_from_left_right_touching_finger_tensor_scene_close_finger(self,fitness_info):
        score_list = [1 if pair[0] and pair[1] else 0 for pair in fitness_info]
        archive_element_list_for_one_tensor= []
        for item_n in range(self.pop_size):
            fitness=score_list[item_n]
            archive_element = {"genome": self.pop_list[item_n], "fitness":fitness ,
                               "behavior_descriptor": self.define_behavior_from_genotype(genotype= self.pop_list[item_n])}
            archive_element_list_for_one_tensor.append(archive_element)
        return  archive_element_list_for_one_tensor

    def affect_fitness_from_difference_int_end_action_joint_tensor_scene_push_action(self, fitness_info):
        norm_l1 = torch.norm(fitness_info, p=1, dim=1) #p : norme1
        fitness_info_list = (norm_l1 > 0).int().tolist()
        archive_element_list = []
        for item_n in range(self.pop_size):
            fitness = fitness_info_list[item_n]
            archive_element = {"genome": self.pop_list[item_n], "fitness": fitness,
                           "behavior_descriptor": self.define_behavior_from_genotype(genotype= self.pop_list[item_n])}
            archive_element_list.append(archive_element)
        return archive_element_list









