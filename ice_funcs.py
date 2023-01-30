"""
FUNCTIONS I GUESS, FILL THIS OUT LATER
"""

import numpy as np
import numba as nb


"""###############################################################
                    Initializer functions
###############################################################"""


@nb.njit
def initialize_sat_map(l, w, initial_sat):
    sat_map = np.zeros((l,w), dtype=np.float64)

    for line in range(l):
        for col in range((l - line + 1)//2):
            sat_map[line, col] = initial_sat

    return sat_map

#### MAYBE USE JIT HERE FOR CONSISTENCY??
def construct_minimal_ice_map(l, w):
    ice_map = np.full((l, w), False, dtype=bool)

    # minimum amount of initial ice that doesn't brick the model
    minimal_ice_coords = np.array([
        [l-1,0],
        [l-2,0],
        [l-3,0],
        [l-3,1]
    ])

    for coords in minimal_ice_coords:
        ice_map[coords[0], coords[1]] = True

    return ice_map


"""###############################################################
                    Neighbor finding utilities
###############################################################"""


@nb.njit
def is_legit_cell(line, col, L):
    return line<L and col<(L - line + 1)//2 and line>=0 and col>=0


@nb.njit
def get_neighbors(line, col, L):
    relative_nearest_neighbors = np.array([
        [-1,0],
        [-1,1],
        [0,1],
        [1,0],
        [1,-1],
        [0,-1]
    ])

    absolute_nearest_neighbors = np.empty((6, 2))

    for i in range(np.shape(relative_nearest_neighbors)[0]):
        neighbor_line = line + relative_nearest_neighbors[i,0]
        neighbor_col = col + relative_nearest_neighbors[i,1]

        if is_legit_cell(neighbor_line, neighbor_col, L):
            absolute_nearest_neighbors[i, :] = [neighbor_line, neighbor_col]
        else:
            absolute_nearest_neighbors[i, :] = [np.nan, np.nan]

    return absolute_nearest_neighbors


@nb.njit
def construct_neighbor_array(l, w):
    neighbor_array = np.empty((l, w, 6, 2))

    for line in range(l):
        for col in range((l - line + 1)//2):
            neighbor_array[line, col, :, :] = get_neighbors(line, col, l)

    return neighbor_array


"""###############################################################
                    Boundary map constructor
###############################################################"""


@nb.njit
def construct_boundary_map(ice_map, neighbor_array):
    l = np.shape(ice_map)[0]
    boundary_map = np.full_like(ice_map, 0, dtype=np.int8)

    for line in range(l):
        for col in range((l - line + 1)//2):
            neighbor_coords = neighbor_array[line, col, :, :] # returns 6x2 array
            
            neighbor_counter = 0

            if ice_map[line, col] != True:
                for i in range(np.shape(neighbor_coords)[0]):
                    neighbor_line = int(neighbor_coords[i,0])
                    neighbor_col = int(neighbor_coords[i,1])

                    if not np.isnan(neighbor_line):
                        if ice_map[neighbor_line, neighbor_col] == True:
                            neighbor_counter += 1

            boundary_map[line, col] = neighbor_counter

    return boundary_map


"""###############################################################
                    Diffusion utilities
###############################################################"""


@nb.njit
def construct_default_diffusion_rules(l, w):
    # diffusion for average-cell
    inner_rule_set = np.ones(6)/6
    
    #diffusion for the upper ragged cells
    upper_ragged_rule_set = np.array([
        1/6,
        1/6,
        0,
        1/6,
        1/6,
        1/3
    ])

    #diffusion for the lower ragged cells
    lower_ragged_rule_set = np.array([
        1/3,
        0,
        0,
        0,
        1/3,
        1/3
    ])

    # diffusion for the flush left cells
    flush_left_rule_set = np.array([
        1/6,
        1/3,
        1/3,
        1/6,
        0,
        0
    ])
    
    diffusion_rules = np.empty((l-1, w-1, 6))

    # fill out cells pertaining to flush left cells
    for line in range(1, l-4):
        diffusion_rules[line, 0, :] = flush_left_rule_set

    # fill out all lower ragged cells
    for col in range(2, w-1):
        line = l - 2*col - 1
        diffusion_rules[line, col, :] = lower_ragged_rule_set

    # fill out all upper ragged cells
    for col in range(2, w-1):
        line = l - 2*col - 2
        diffusion_rules[line, col, :] = upper_ragged_rule_set

    # fill out the rest
    for col in range(1, w-1):
        for line in range(1, l - 2*col - 2):
            diffusion_rules[line, col, :] = inner_rule_set

    return diffusion_rules


@nb.njit
def diffuse_cell(old_sat_map, local_diffusion_rules, neighbors):
    new_sat = 0 # initialize new saturation 
    
    for i in range(np.shape(neighbors)[0]):
        line = int(neighbors[i,0])
        col = int(neighbors[i,1])

        new_sat += old_sat_map[line, col]*local_diffusion_rules[i]
        
    return new_sat


@nb.njit
def relax_sat_map(old_sat_map, diffusion_rules, ice_map, boundary_array, neighbor_array, l):
    new_sat_map = old_sat_map.copy()

    for line in range(1,l):
        for col in range((line+1)//2):
            # Checks if the cell is not an 'ice cell' or a 'boundary cell'
            if ice_map[line, col] == False and boundary_array[line, col] == 0:
                new_sat_map[line, col] = diffuse_cell(
                    old_sat_map, 
                    diffusion_rules[line, col, :], 
                    neighbor_array[line, col, :, :]
                )


    return new_sat_map


@nb.njit
def has_converged(sat_1, sat_2, epsilon):
    array_shape = np.shape(sat_1)

    for i in range(array_shape[0]):
        for j in range(array_shape[1]):
            if abs(sat_1[i,j] - sat_2[i,j]) > epsilon:
                return False

    return True


# def