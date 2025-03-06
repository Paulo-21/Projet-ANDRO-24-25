import pdb

import torch

u_vect = torch.tensor([[ 0.0000e+00,  0.0000e+00,  1.0000e+00],
        [ 0.0000e+00,  0.0000e+00,  1.0000e+00],
        [ 6.1676e-04, -1.4901e-07,  1.0000e+00],
        [-7.1526e-07, -1.1921e-07,  1.0000e+00]], device='cuda:0')

theta = torch.tensor([[1],
        [ 2],
        [3],
        [ 4]], device='cuda:0')

def compute_tensor(theta, vectors):

    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]  # Décomposer les coordonnées
    theta_squeezed = theta.squeeze(-1)
    c = torch.cos(theta_squeezed)
    s = torch.sin(theta_squeezed)

    """
    # Construire la matrice demandée
    matrix = torch.stack([torch.stack([theta,x,x], dim=-1),
                                torch.stack([x, y, z], dim=-1),
                                torch.stack([0.5 * x, 0.5 * y, 0.5 * z], dim=-1)],
                             dim=1)  # [x**2*(1-x)**2*theta, x*y*(1-theta)-z*theta, x*z*(1-theta)+y*theta]
    pdb.set_trace()
                            
    """
    matrix = torch.stack([torch.stack([x**2+(1-x**2)*c, x*y*(1-c)-z*s, x*z*(1-c)+y*s], dim=-1),
                          torch.stack([x*y*(1-c)+z*s, y**2*(1-y**2)*c, y*z*(1-c)-x*s], dim=-1),
                          torch.stack([x*y*(1-c)-y*s, y*z*(1-c)+x*s, z**2+(1-z**2)*c], dim=-1)],
                         dim=1)
    pdb.set_trace()



    return matrix

# Exemple d'utilisation
vectors = torch.tensor([[1, 2, 3],
                        [4, 5, 6]], dtype=torch.float32, device='cuda:0')  # Vecteurs (2x3)

result = compute_tensor(theta, u_vect)

# Afficher le résultat
print(result)
pdb.set_trace()
