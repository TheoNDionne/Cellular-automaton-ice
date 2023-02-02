""" snowflake_plot.py

A suite of functions useful for plotting snowflakes using the 'ice_map' array.
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, pi


@nb.njit
def coords_from_array_position(line, col, L):
    # translate (line, col) to (x,y) with origin at (line, col) = (L-1,0)
    array_x = col
    array_y = L - line - 1

    # make cartesian coordinates
    x_coord = sqrt(3)*array_x/2
    y_coord = array_y - 0.5*array_x

    return x_coord, y_coord


@nb.njit
def construct_coords_from_ice_map(ice_map, L):
    x_coords = []
    y_coords = []

    for line in range(L):
        if ice_map[line, 0] == True:
            x, y = coords_from_array_position(line, 0, L) 

            x_coords.append(x)
            y_coords.append(y)

    for col in range(1, (L+1)//2):
        for line in range(L-2*col):
            if ice_map[line, col] == True:
                x, y = coords_from_array_position(line, col, L) 

                # append cell with positive x
                x_coords.append(x)
                y_coords.append(y)

                # append cell with negative x
                x_coords.append(-x)
                y_coords.append(y)

    return np.transpose(np.array([x_coords, y_coords]))


@nb.njit
def rotate_coordinates(vectors, angle):
    cosine = cos(angle)
    sine = sin(angle)

    vectors_shape = np.shape(vectors)
    new_vectors = np.empty(vectors_shape)
    
    for i in range(vectors_shape[0]):
        x = vectors[i,0]
        y = vectors[i,1]

        new_vectors[i,0] = cosine*x - sine*y 
        new_vectors[i,1] = sine*x + cosine*y

    return new_vectors


@nb.njit
def reconstruct_all_coordinates(vectors):
    angles = 2*pi*np.arange(60, 361, 60)/360
    number_of_vectors = np.shape(vectors)[0]
    
    all_coordinates = np.empty((number_of_vectors*6, 2))
    all_coordinates[:number_of_vectors, :] = vectors

    for i in range(5):
        all_coordinates[(i+1)*number_of_vectors:(i+2)*number_of_vectors, :] = rotate_coordinates(vectors, angles[i])

    return all_coordinates


@nb.njit
def convert_ice_map_to_all_coords(ice_map, L):
    upper_sixth_positions = construct_coords_from_ice_map(ice_map, L)
    all_positions = reconstruct_all_coordinates(upper_sixth_positions)

    return all_positions


#### implement animation and things (rework vector things...)

#### IMPLEMENT IN THE ANIMATION AS A SUBSTRACTIVE ice_map' - ice_map to reduce calculation steps (process is irreversible) [via XOR '^']