from dynamic_map_qd import Archive
print("dyna")

archive_element = {"genome": [0.3333333, 0.21, 0.93], "fitness": 80, "behavior_descriptor" : [0.3, 0.2, 0.9]}
archive_element2= {"genome": [0.3222222, 0.21, 0.93], "fitness": 80, "behavior_descriptor" : [0.3, 0.2, 0.9]}
archive_element3= {"genome": [0.3122, 0.23, 0.94], "fitness": 80,    "behavior_descriptor" : [0.3, 0.2, 0.9]}
x = Archive("Close finger")
#x.set_max_depth(2)
x.set_max_precision([0.001, 0.001, 0.001])
"""
x.store_one_new_element_in_archive(archive_element)
x.store_one_new_element_in_archive(archive_element2)
x.store_one_new_element_in_archive(archive_element3)
"""
x.store_several_element_in_archive([archive_element, archive_element2, archive_element3])
selected = x.select_random_individuals_from_archive(10)
print(selected)