import csv
import pdb
import numpy as np
BOOTSTRAP_DICTIONARY_HANDMADE = {
    "10211": np.array([[0.036, - 0.04514433, 0.12502763, 0.70341268, 0.2010352, 0.78229176, 0.13505072]]),
    "100658" : np.array([[-0.432, 0.674, 0.427, 0.494, 0.849, 0.180, -0.060] ]),
    "9032" : np.array([[-1.87310494,  1.23429593,  1.12930435,  0.37725285,  0.04581762,  0.9562257, -0.75245328]]),
    "10143" : np.array([[-0.534, 0.234, 0.205, 0.430, 0.536, 0.455, 0.567]]),
    "7310" : np.array([[-0.733, -0.309, 0.052, 0.059, 0.720, -0.026, 0.691]]) #microwave
}

#bootstrap pour carton :
#basic np.array([[0.8, 0 , 0.7,1.7948965149208059e-09, 1.0, 0.0, 0.0]])

# (joint 4 carton) [-0.376, -0.661, 0.402, -0.189, -0.089, 0.824, 0.526]
# (joint 2 carton) : [-0.432, 0.674, 0.427, 0.494, 0.849, 0.180, -0.060]
# (joint 1 carton) [0.930,-0.267,0.468, -0.404, 0.549, 0.663, -0.309]
# (joint3 carton) [-0.944,0.234,0.412,0.335,0.632,0.489,0.499] (peti cote) ou  [-0.921, -0.226, 0.459, 0.405, 0.522,0.655,0.258]

def convert_to_list(string):
    # Retirer les crochets et découper en éléments non vides
    string = string.strip('[]').split()
    # Convertir chaque élément en float
    python_list = [float(x) for x in string]
    return python_list

def read_successful_grasp(path):
    individuals = []
    with open(path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            individual = convert_to_list(row[0])
            if row[1]=='0' :
               success = False
            else:
               success = True
               individuals.append(individual)
    return individuals
