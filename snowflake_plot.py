""" snowflake_plot.py

A suite of functions useful for plotting snowflakes using the 'ice_map' array.
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, pi


@nb.njit
def coords_from_array_position(line, col, L):
    """ Function that converts array positions (line, col) to 
    pseudo-cartesian coordinates with an origin at cell (L-1,0).

    Arguments
    ---------
    line : int
        The line of the ice_cell in ice_map.
    col : int
        The column of the ice_cell in ice_map.
    L : int
        The 'length' of the simulation zone, in units of 'cells'.

    Return
    ------
    (tuple) Returns the x coordinate and then the y coordinate of
     specified cell as (x,y).
    """
    # translate (line, col) to (x,y) with origin at (line, col) = (L-1,0)
    array_x = col
    array_y = L - line - 1

    # make cartesian coordinates
    x_coord = sqrt(3)*array_x/2
    y_coord = array_y - 0.5*array_x

    return x_coord, y_coord # coordinate tuple


@nb.njit
def construct_coords_from_ice_map(ice_map, L):
    """ Function that reads over a boolean 'ice map' and creates 
    a pair of coordinates for every ice cell and returns coordinates
    as an array of coordinates. Automatically applies the reflexion 
    boundary condition that exists along the edge defined by col=0.

    Arguments
    ---------
    ice_map : array(bool, 2d)
        Boolean map of ice cells (True if ice, False or else).
    L : int
        The 'length' of the simulation zone, in units of 'cells'.

    Return
    ------
    (array(float, shape=(:,2)))
    """
    # initialize coordinate lists
    x_coords = []
    y_coords = []

    # translate and add coords from col=0
    for line in range(L):
        if ice_map[line, 0] == True: # checks if ice cell
            x, y = coords_from_array_position(line, 0, L) 

            x_coords.append(x)
            y_coords.append(y)

    # translate and add all other ice cell coords and their reflexion about x=0
    for col in range(1, (L+1)//2):
        for line in range(L-2*col):
            if ice_map[line, col] == True: # checks if ice cell
                x, y = coords_from_array_position(line, col, L) 

                # append cell with positive x
                x_coords.append(x)
                y_coords.append(y)

                # append cell with negative x (reflexion)
                x_coords.append(-x)
                y_coords.append(y)

    return np.transpose(np.array([x_coords, y_coords])) # returns array of ice cell coordinates


@nb.njit
def rotate_coordinate_array(vectors, angle):
    """ A function that takes an array of cartesian coordinate 
    vectors and rotates them by the specified polar angle (in radians).

    Arguments
    ---------
    vectors : array(float, 2d)
        Array of cartesian vector coordinates in the shape produced by 
        construct_coords_from_ice_map().
    angle : float
        Angle (in radians) by which to rotate position of ice cells in 
        the conventionnal direction.

    Return
    ------
    (array(float, 2d)) Array of vector coordinates post-rotation. 
    """
    # Calculate trig functions for rotation transfo
    cosine = cos(angle) 
    sine = sin(angle)

    vectors_shape = np.shape(vectors)
    new_vectors = np.empty(vectors_shape) # initialize new vector array
    
    # execute rotation step-by-step (haha) for numba compatibility
    for i in range(vectors_shape[0]):
        # fetch current coordinates
        x = vectors[i,0] 
        y = vectors[i,1]

        # rotate coordinates
        new_vectors[i,0] = cosine*x - sine*y 
        new_vectors[i,1] = sine*x + cosine*y

    return new_vectors


@nb.njit
def reconstruct_all_coordinates(vectors):
    """ A function that takes all the vector coordinates of the ice cells 
    in the upper sixth of the hexagonal mesh (i.e. the reflected simulation 
    zone) and rotates it to fill entire hexagonal grid.

    Arguments
    ---------
    vectors : array(float, 2d)
        Array of cartesian vector coordinates in the shape produced by 
        construct_coords_from_ice_map().

    Return
    ------
    (array(float, 2d)) Array of cartesian vector coordinates for all cells 
    in the shape produced by construct_coords_from_ice_map().   
    """
    angles = 2*pi*np.arange(60, 361, 60)/360 # generates list of rotation angles in radians (1/6th of a turn 5 times)
    number_of_vectors = np.shape(vectors)[0] # amount of the 
    
    all_coordinates = np.empty((number_of_vectors*6, 2)) # initialize total position vector array
    all_coordinates[:number_of_vectors, :] = vectors # initialize first sixth of the 

    # Add rotated cell coords to total coords array
    for i in range(5):
        all_coordinates[(i+1)*number_of_vectors:(i+2)*number_of_vectors, :] = rotate_coordinate_array(vectors, angles[i])

    return all_coordinates


@nb.njit
def convert_ice_map_to_all_coords(ice_map, L):
    """ A wrapper function that converts the provided ice map 
    to an array of cartesian coordinates representing the center 
    positions of all cells.

    Arguments
    ---------
    ice_map : array(bool, 2d)
        Boolean map of ice cells (True if ice, False or else).
    L : int
        The 'length' of the simulation zone, in units of 'cells'.

    Return
    ------
    (array(float, 2d)) Array of cartesian vector coordinates for all cells 
    in the shape produced by construct_coords_from_ice_map().   
    """
    upper_sixth_positions = construct_coords_from_ice_map(ice_map, L) # generate positions for the upper sixth
    all_positions = reconstruct_all_coordinates(upper_sixth_positions) # turn sixth into entire coordinates for all cells

    return all_positions


#### implement animation and things (rework vector things...)

#### IMPLEMENT IN THE ANIMATION AS A SUBSTRACTIVE ice_map' - ice_map to reduce calculation steps (process is irreversible) [via XOR '^']