"""
Functions used to simulate the formation of a C6v symmetric snowflakes.

*** Re-implementation of bits of code found in numpy are due to ***
*** restricted numpy support in numba.                          ***

"""

import numpy as np
import numba as nb


"""###############################################################
                    Physics-based functions
###############################################################"""


def attachment_coefficient_with_kink(sat, kink, max_amplitude=1):
    max_amplitude*np.exp(-kink*sat) ############################## NOT SURE THIS IS OK


"""###############################################################
                    Initializer functions
###############################################################"""


@nb.njit
def initialize_sat_map(l, w, initial_sat):
    """ Function that initializes the saturation map at a 
    constant value `initial_sat`.

    Arguments
    ---------
    l : int
        Total length of the restricted simulation zone (1/12th) 
        of total snowflake.
    w : int
        Total width of the restricted simulation zone.

    Return
    ------
    The initialized saturation map. (array(float, 2d))
    """ 
    sat_map = np.zeros((l,w), dtype=np.float64)

    for line in range(l):
        for col in range((l - line + 1)//2):
            sat_map[line, col] = initial_sat

    return sat_map


#### MAYBE USE JIT HERE FOR CONSISTENCY??
def construct_minimal_ice_map(l, w):
    """ Function that constructs the minimal ice map defined 
    in the model (4 cells).

    Arguments
    ---------
    l : int
        Total length of the restricted simulation zone (1/12th) 
        of total snowflake.
    w : int
        Total width of the restricted simulation zone.
    
    Return
    ------
    The minimal possible ice map. (array(bool, 2d))
    """
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
    """ Function that verified that provided coordinates are 
    part of the model.
    
    Arguments
    ---------
    line : int
        The line of the cell to check in the model.
    col : int
        The column of the cell to check in the model.

    Return
    ------
    Returns if the cell is a legit simulation cell. (bool)
    """
    return line<L and col<(L - line + 1)//2 and line>=0 and col>=0


@nb.njit
def get_neighbors(line, col, l):
    """ Function that gets nearest neighbors of a hexagonal cell
    in the simulation model.

    Arguments
    ---------
    line : int
        The line of the cell to check in the model.
    col : int
        The column of the cell to check in the model.
    L : int
        Total length of the model. 

    Return
    ------
    The neighbors of cell at position (line, col) in the model (array(float, 2d)).
    If there is no neighbor, coords are replaced by `np.nan`. Array of the form 
    Arr[k,l], where:
    - k : 'neighbor index' (6 possibilities for hexagonal cells).
    - l : toggle between line and column of neighbor at index `k`.
    """
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

        if is_legit_cell(neighbor_line, neighbor_col, l):
            absolute_nearest_neighbors[i, :] = [neighbor_line, neighbor_col]
        else:
            absolute_nearest_neighbors[i, :] = [np.nan, np.nan]

    return absolute_nearest_neighbors


@nb.njit
def construct_neighbor_array(l, w):
    """ Constructs the array of nearest neighbors for all cells in the model.

    Arguments
    ---------
    l : int
        Total length of the restricted simulation zone (1/12th) 
        of total snowflake.
    w : int
        Total width of the restricted simulation zone. 

    Return
    ------
    Returns a 4d array of neighbor cells for all cells (array(float, 4d)).
    Array is indexed with the following indexes Arr[i,j,k,l], where:
    - i : line of the cell for which to get neighbors.
    - j : column of the cell for which to get neighbors.
    - k : 'neighbor index' (6 possibilities for hexagonal cells).
    - l : toggle between line and column of neighbor at index `k`.
    """
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
    """ Function that constructs a map of all boudary pixels. The number 
    ascribed to each cell is the number of nearest neighbors (AKA: the kink neighbor).

    Arguments
    ---------
    ice_map : array(bool, 2d)
        The boolean map that represents the physical location of ice cells.
    neighbor_array : array(float, 4d)
        The array representing neigbors in the model of the form produced by 
        construct_neighbor_array().
    
    Return
    ------
    The two dimensional map of the amount of neighboring ice cells (excluding 
    ice cells themselves). (array(int, 2d))
    """
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
    """ Constructs the 'vanilla' diffusion rules using the discretized
    version of the diffusion equation.

    Arguments
    ---------
    l : int
        Total length of the restricted simulation zone (1/12th) 
        of total snowflake.
    w : int
        Total width of the restricted simulation zone. 

    Retour
    ------
    Returns the set of diffusion rules for the model (array(float, 3d)).
    Array is indexed as Arr[i,j,k], where:
    - i : line of the cell to diffuse.
    - j : column of the cell to diffuse.
    - k : coefficient by which to multiply saturation situated at 
          neighbor index `k`.
    """
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
    
    diffusion_rules = np.empty((l, w, 6))

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
def diffuse_cell(sat_map, local_diffusion_rules, neighbors):
    """ Returns the value of a cell after one step of the diffusion process.

    Arguments
    ---------
    sat_map : array(float, 2d)
        The saturation field before the diffusion step.
    local_diffusion_rules : array(float, 1d)
        The local diffusion rules as a slice of diffusion_rules[i,j,:].
    neighbors : array(float, 2d)
        The local neighbors as a slice of neighbor_array[i,j,:,:].

    Return
    ------
    The new sat map after a diffusion iteration. (array(float, 2d))
    """
    new_sat = 0 # initialize new saturation 
    
    for i in range(np.shape(neighbors)[0]):
        line = int(neighbors[i,0])
        col = int(neighbors[i,1])

        new_sat += sat_map[line, col]*local_diffusion_rules[i]
        
    return new_sat


# @nb.njit
# def relax_sat_map(old_sat_map, diffusion_rules, ice_map, boundary_array, neighbor_array, l):
#     """ JESUS CHRIST FIGURE THIS OUT LMAO.

#     """
#     new_sat_map = old_sat_map.copy()

#     for line in range(1,l):
#         for col in range((line+1)//2):
#             # Checks if the cell is not an 'ice cell' or a 'boundary cell'
#             if ice_map[line, col] == False and boundary_array[line, col] == 0:
#                 new_sat_map[line, col] = diffuse_cell(
#                     old_sat_map, 
#                     diffusion_rules[line, col, :], 
#                     neighbor_array[line, col, :, :]
#                 )


#     return new_sat_map


# This needed to be coded since njit does not yet support np.allclose or np.isclose
@nb.njit
def has_converged(sat_1, sat_2, epsilon):
    """ Function that allows for an element-wise convergence assessment
    of two arrays of the form |a-b| < epsilon.

    Arguments
    ---------
    sat_1 : array(float, 2d)
        The first array to be compared.
    sat_2 : array(float, 2d)
        The second array to be compared.
    epsilon : float
        The maximum allowed difference in the convergence criterion.

    Return
    ------
    The truth status of the convergence: `True` if all element-wise 
    differences are smaller than epsilon, `False` or else. (bool)
    """
    array_shape = np.shape(sat_1)

    for i in range(array_shape[0]):
        for j in range(array_shape[1]):
            if abs(sat_1[i,j] - sat_2[i,j]) > epsilon:
                return False

    return True


@nb.njit
def calculate_sat_opp(sat_map, ice_map, local_neighbors):
    opposing_indices = np.array([3,4,5,0,1,2]) # in 'neighbor index' form
    opp_cell_counter = 0
    opp_cell_total = 0.0

    for i in range(np.shape(local_neighbors)):
        neighbor_line = local_neighbors[i,0]
        neighbor_col = local_neighbors[i,1]

        opp_line = local_neighbors[opposing_indices[i],0]
        opp_col = local_neighbors[opposing_indices[i],1]

        if ice_map[neighbor_line, neighbor_col] == True and ice_map[opp_line, opp_col] == False:
            opp_cell_total += sat_map[opp_line, opp_col]
            opp_cell_counter += 1

    return opp_cell_total / opp_cell_counter # returns the average of the 'opposing' cells        


# def apply_boundary_condition():
#     # sigma = sigma_opp/(1+ alpha(sigma_old)G_b*Dx/X_0)

def get_opp_neighbor_indices(ice_map, local_neighbors):
    opposing_indices = np.array([3,4,5,0,1,2]) # in 'neighbor index' form
    local_opp_array = np.full(3, np.nan)

    opps_detected = 0
    for i in range(np.shape(local_neighbors)[0]):
        neighbor_line = local_neighbors[i,0]
        neighbor_col = local_neighbors[i,1]

        opp_line = local_neighbors[opposing_indices[i],0]
        opp_col = local_neighbors[opposing_indices[i],1]

        if ice_map[neighbor_line, neighbor_col] == True and ice_map[opp_line, opp_col] == False:
            local_opp_array[opps_detected] = opposing_indices[i] # adds opposing 'neighbor index'
            opps_detected += 1

    return local_opp_array


def distinguish_cells(ice_map, boundary_map, l):
    X_normal = []
    Y_normal = []
    
    X_boundary = []
    Y_boundary = []

    for line in range(1,l):
        for col in range((line+1)//2):
            if ice_map[line, col]:
                if boundary_map[line, col] == 0:
                    X_normal.append(line)
                    Y_normal.append(col)
                else:
                    X_boundary.append(line)
                    Y_boundary.append(col)

    normal_cells = np.transpose(np.array([X_normal, Y_normal]))
    boundary_cells = np.transpose(np.array([X_boundary, Y_boundary]))
    
    return normal_cells, boundary_cells


def construct_opp_array(boundary_cells, ice_map, neighbor_array):
    array_length = np.shape(boundary_cells)[0]
    opp_array = np.empty((array_length, 3)) 
    
    for i in range(array_length):
        line = int(boundary_cells[i,0])
        col = int(boundary_cells[i,1])
        local_neighbors = neighbor_array[line, col, :, :]

        opp_array[i,:] = get_opp_neighbor_indices(ice_map, local_neighbors)

    return opp_array


def calculate_sat_opp_average(sat_map, local_neighbors, local_opps):
    sat_sum = 0
    opp_counter = 0

    # loops over the 'opposing' cells
    for neighbor_index in local_opps:
        if not np.isnan(neighbor_index):
            opp_counter += 1 # increments the total count of opps
            opp_line = local_neighbors[neighbor_index, 0]
            opp_col = local_neighbors[neighbor_index, 1]

            sat_sum += sat_map[opp_line, opp_col]

    return sat_sum / opp_counter # returns the average of opposing saturations


def apply_boundary_condition(line, col, sat_map, local_neighbors, local_opps, boundary_map, D_x, G_b=1, X_0=1):
    local_sat = sat_map[line, col]
    kink_number = boundary_map[line, col]
    sat_opp = calculate_sat_opp_average(sat_map, local_neighbors, local_opps)

    alpha = attachment_coefficient_with_kink(local_sat, kink_number)

    return sat_opp/(1 + alpha*G_b*D_x/X_0) # returns the new saturation of the boundary cell


def execute_relaxation_step(old_sat_map, normal_cells, boundary_cells, opp_array, neighbor_array, diffusion_rules):
    """ JESUS CHRIST FIGURE THIS OUT LMAO.

    """
    new_sat_map = old_sat_map.copy()

    normal_cell_amount = np.shape(normal_cells)[0]
    boundary_cell_amount = np.shape(boundary_cells)[0]

    # diffuse all normal cells previously found
    for i in range(normal_cell_amount):
        line = int(normal_cells[i,0])
        col = int(normal_cells[i,1])

        local_neighbors = neighbor_array[line, col, :, :]
        local_diffusion_rules = diffusion_rules[line, col, :]

        new_sat_map[line, col] = diffuse_cell(old_sat_map, local_diffusion_rules, local_neighbors)

    # application of boundary conditions to 
    for i in range(boundary_cell_amount):
        line = int(normal_cells[i,0])
        col = int(normal_cells[i,1])

        local_neighbors = neighbor_array[line, col, :, :]
        local_diffusion_rules = diffusion_rules[line, col, :]
        
        pass # Apply boundary conditions

    return new_sat_map


# Think of growth steps