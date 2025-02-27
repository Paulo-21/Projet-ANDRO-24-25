import pdb

import torch

u_vect = torch.tensor([[ 0.0000e+00,  0.0000e+00,  1.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  1.0000e+00],
        [ 6.1676e-04, -1.4901e-07,  1.0000e+00],
        [-7.1526e-07, -1.1921e-07,  1.0000e+00]], device='cuda:0')


def compute_tensor(theta, vectors):
    """
    Calcule la matrice demandée en appliquant theta sur chaque élément du tenseur de vecteurs.

    Args:
        theta (torch.Tensor): Scalaire unique (ex: theta2)
        vectors (torch.Tensor): Tenseur de forme (N, 3) où chaque ligne est un vecteur [x, y, z]

    Returns:
        torch.Tensor: Tenseur de forme (N, 2, 3) avec les transformations appliquées
    """
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]  # Décomposer les coordonnées

    # Construire la matrice demandée
    matrix = torch.stack(   [ torch.stack([x**2, y**2, z**2], dim=-1) ,
                            torch.stack([x, y, z], dim=-1) ,
                            torch.stack([0.5*x, 0.5*y, 0.5*z], dim=-1)],
                            dim=1)#[x**2*(1-x)**2*theta, x*y*(1-theta)-z*theta, x*z*(1-theta)+y*theta]


    return matrix

# Exemple d'utilisation
theta2 = torch.tensor(2.0, dtype=torch.float32, device='cuda:0')  # Scalaire theta2
vectors = torch.tensor([[1, 2, 3],
                        [4, 5, 6]], dtype=torch.float32, device='cuda:0')  # Vecteurs (2x3)

result = compute_tensor(theta2, u_vect)

# Afficher le résultat
print(result)
pdb.set_trace()
