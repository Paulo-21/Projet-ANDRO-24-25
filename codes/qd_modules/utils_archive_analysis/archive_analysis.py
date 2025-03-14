import pandas as pd
class ArchiveAnalysis():
    def __init__(self, name_archive):
        self.name_archive = name_archive
        self.archive_map =None
        self.data_frame = pd.read_csv(name_archive)

    def plot_sphere_on_scene(self, simulator_scene_simualtion):
        my_QD_algo.simulator_scene_simualtion

