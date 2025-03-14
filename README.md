# QD_action_project



## Prerequisites : 
sudo singularity build --sandbox singularity_project/singularity_mode.sif singularity_project/singularity_mode.def 
sudo singularity shell --writable singularity_project/singularity_mode.sif



sudo singularity shell --writable --bind /tmp/.X11-unix:/tmp/.X11-unix,result_archive:/result_archive,results:/results --env DISPLAY=$DISPLAY singularity_project/singularity_mode2.sif
Singularity> python3 /codes/main_map_elite_action.py 

OPENGL
`apt-get update
apt-get install libgl1-mesa-glx
apt-get update
apt-get install libglib2.0-0`

SAPIEN is a realistic and physics-rich simulated environment. Compatible with Linux: Ubuntu 18.04+, No GPU needed. 

Git Clone the reference folder for this project :
`git clone .....`

In the reference folder, create your virtual environment with Python3

`python -m venv /path/to/new/virtual/my_sapien_env`

Activate your virtual environment:

`source my_sapien_env/bin/activate`

Install via Sapien via Pip in your python3.12 environment :

`pip install sapien`

run this command :

`pip install --upgrade setuptools`

Verify the installation by running :

`python -m sapien.example.hello_world`

This command should open a viewer window showing a red cube on the ground.

## Run the reference code on your computer: 


Run the code by running the following command in your virtual environment. 
`python3 main_map_elite_action.py`

**QD_parameters**
- ```biased_selection``` : if _False_ simple Map-Elite is implemented (based on a random selection 
of individuals in the archive). If _True_ Map_Elite_success is implemented( based on a **biased selection**
of individuals in the archive)
- ```nb_generations=nb_generations``` : ex 300,
- ```coefxyz_mutation``` : QD mutation coefficient, 

- ```pop_size=pop_size``` : ex 10 or 100,
- ```bootstrap``` : if _True_ the algo initializes population with given individuals 

**general algo parameter**

- ```dynamic_application``` : ex : "close_finger" , "push_forward" , "push_left"
- ```object_to_grasp``` : Number of the object to grasp  "100658" for the cardboard
- ```bb```= Bonding box around the object, range of the genotype space for x, y and z
- ```multi_thread=multi_thread``` : if _True_, algo in multithreading. Use _False_ for debugging
- ```render_mode=render_mode``` : Diplay the 3D scene in your computer,
- ```generation_mode``` : if _True_ the algo generates new solutions, if _False_ the algo read solutions previously stored in a csv file,
- ```gripper``` : ex : "panda" end effector used, for now only panda is available
- ```artificial_bb```= artificial bounding box 
- ```replay_indx``` : replay only one individual, several time
- ```version``` : version of the csv file to open while reading, if no number : None ,


## Pistes d ameliorations: 

- Afficher les couvertures d archives (stockees dans des csv pour l instant) sur des graphes matplotlib
- regeler les coefficients de QD, notemment echelle de mutation
- - Coder le bootstrap d archive il faut amorcer les pushmode sur le contact mode

- Mettre dans le genome stiffness and damping, changer l espace des comportements. 
## Using container Singularity
Build the container from existing def file :
`sudo singularity build singularity_project/singularity_mode.sif singularity_project/singularity_mode.def `
Then 
`xhost +local:root`

run the container
` sudo singularity shell --bind /tmp/.X11-unix:/tmp/.X11-unix,`result_archive:/result_archive,results:/results` --env DISPLAY=$DISPLAY singularity_project/singularity_mode.sif`
Then inside the container run the following command : 
`cd /`

`sudo singularity build --sandbox singularity_project/singularity_mode.sif singularity_project/singularity_mode.def `

`singularity exec --pwd / --bind result_archive:/result_archive,results:/results singularity_project/singularity_mode.sif python3 codes/main_map_elite_action.py`

## Verison 

La scene met du temps a builder la prmiere fois qu on la build, apres c est beaucoup plus ra# quaternions_genesis
python3 codes/main_map_elite_action.py --robot 'entire_robot' --obj_name "carton" --action "close_finger" --multi_thread "GPU_parallel" --viz 'False'  --render_mode 'False'
python3 codes/main_map_elite_action.py --robot 'entire_robot' --obj_name "carton" --action "close_finger" --multi_thread "GPU_parallel" --viz 'False'  --render_mode 'False'
