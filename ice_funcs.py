"""
Functions used to simulate the formation of a C6v symmetric snowflakes.

*** Re-implementation of bits of code found in numpy are due to ***
*** restricted numpy support in numba.                          ***

NOTE: To keep track of a hexagonal grid with rectangular arrays, one 
must come up with a mapping that preserves nearest neighbors. The mapping 
used in these functions is the same as the one used by K.G. Librecht in
Snow Crystals, i.e.:

         _____        
        /     \                            _______ _______ _______
  _____/   0   \_____                     |       |       |       |                   
 /     \       /     \                    |       |   0   |   1   | 
/   5   \_____/   1   \                   |_______|_______|_______|
\       /     \       /                   |       |       |       |
 \_____/   X   \_____/         ====>      |   5   |   X   |   2   |
 /     \       /     \                    |_______|_______|_______|                       
/   4   \_____/   2   \                   |       |       |       |
\       /     \       /                   |   4   |   3   |       |
 \_____/   3   \_____/                    |_______|_______|_______|
       \       /
        \_____/

The number present in the cells indicate the so-called "neighbor index" 
relative to the central cell denoted by "X".

"""

import numpy as np
import numba as nb


"""###############################################################
                    Initializer functions
###############################################################"""


@nb.njit
def initialize_sat_map(l, w, initial_sat=1):
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

    return np.full((l,w), initial_sat, dtype=np.float64)


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
    ice_map = np.full((l, w), False, dtype=bool) # initialize 'empty' ice_map

    # minimum amount of initial ice that doesn't brick the model
    minimal_ice_coords = np.array([
        [l-1,0],
        [l-2,0],
        [l-3,0],
        [l-3,1]
    ])

    for coords in minimal_ice_coords:
        ice_map[coords[0], coords[1]] = True # sets specified cells to be ice

    return ice_map


"""###############################################################
                    Neighbor finding utilities
###############################################################"""

NeighborUtilities_spec = {
    "L" : nb.int32,
    "W" : nb.int32
}

@nb.experimental.jitclass(NeighborUtilities_spec)
class NeighborUtilities:
    """ Class that allows to     
    """

    def __init__(self, L):
        """ Class initializer function.
        
        Attributes
        ----------
        L : int
            total length of the 1/12th slice of simulation zone.
        W : int
            total width of the 1/12th slice of simulation zone.
        """
        self.L = L
        self.W = (self.L+1)//2

    def is_legit_cell(self, line, col):
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
        return line<self.L and col<(self.L - line + 1)//2 and line>=0 and col>=0

    def get_neighbors(self, line, col):
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
        # relative positions of nearest neighbors in grid
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

            if self.is_legit_cell(neighbor_line, neighbor_col):
                absolute_nearest_neighbors[i, :] = [neighbor_line, neighbor_col]
            else:
                absolute_nearest_neighbors[i, :] = [np.nan, np.nan]

        return absolute_nearest_neighbors

    def construct_neighbor_array(self):
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
        neighbor_array = np.empty((self.L, self.W, 6, 2))

        for line in range(self.L):
            for col in range((self.L - line + 1)//2):
                neighbor_array[line, col, :, :] = self.get_neighbors(line, col)

        return neighbor_array


# @nb.njit
# def is_legit_cell(line, col, L):
#     """ Function that verified that provided coordinates are 
#     part of the model.
    
#     Arguments
#     ---------
#     line : int
#         The line of the cell to check in the model.
#     col : int
#         The column of the cell to check in the model.

#     Return
#     ------
#     Returns if the cell is a legit simulation cell. (bool)
#     """
#     return line<L and col<(L - line + 1)//2 and line>=0 and col>=0


# @nb.njit
# def get_neighbors(line, col, l):
#     """ Function that gets nearest neighbors of a hexagonal cell
#     in the simulation model.

#     Arguments
#     ---------
#     line : int
#         The line of the cell to check in the model.
#     col : int
#         The column of the cell to check in the model.
#     L : int
#         Total length of the model. 

#     Return
#     ------
#     The neighbors of cell at position (line, col) in the model (array(float, 2d)).
#     If there is no neighbor, coords are replaced by `np.nan`. Array of the form 
#     Arr[k,l], where:
#     - k : 'neighbor index' (6 possibilities for hexagonal cells).
#     - l : toggle between line and column of neighbor at index `k`.
#     """
#     # relative positions of nearest neighbors in grid
#     relative_nearest_neighbors = np.array([
#         [-1,0],
#         [-1,1],
#         [0,1],
#         [1,0],
#         [1,-1],
#         [0,-1]
#     ])

#     absolute_nearest_neighbors = np.empty((6, 2))

#     for i in range(np.shape(relative_nearest_neighbors)[0]):
#         neighbor_line = line + relative_nearest_neighbors[i,0]
#         neighbor_col = col + relative_nearest_neighbors[i,1]

#         if is_legit_cell(neighbor_line, neighbor_col, l):
#             absolute_nearest_neighbors[i, :] = [neighbor_line, neighbor_col]
#         else:
#             absolute_nearest_neighbors[i, :] = [np.nan, np.nan]

#     return absolute_nearest_neighbors


# @nb.njit
# def construct_neighbor_array(l, w):
#     """ Constructs the array of nearest neighbors for all cells in the model.

#     Arguments
#     ---------
#     l : int
#         Total length of the restricted simulation zone (1/12th) 
#         of total snowflake.
#     w : int
#         Total width of the restricted simulation zone. 

#     Return
#     ------
#     Returns a 4d array of neighbor cells for all cells (array(float, 4d)).
#     Array is indexed with the following indexes Arr[i,j,k,l], where:
#     - i : line of the cell for which to get neighbors.
#     - j : column of the cell for which to get neighbors.
#     - k : 'neighbor index' (6 possibilities for hexagonal cells).
#     - l : toggle between line and column of neighbor at index `k`.
#     """
#     neighbor_array = np.empty((l, w, 6, 2))

#     for line in range(l):
#         for col in range((l - line + 1)//2):
#             neighbor_array[line, col, :, :] = get_neighbors(line, col, l)

#     return neighbor_array


"""###############################################################
                    Boundary map constructor
###############################################################"""


@nb.njit
def construct_boundary_map(ice_map, neighbor_array):
    """ Function that constructs a map of all boudary pixels. The number 
    ascribed to each cell is the number of nearest neighbors (AKA: the kink number).

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
    boundary_map = np.full((l, (l+1)//2), 0)

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
                    Physics-based functions
            (all based on [arXiv:1910.06389, chap.5])
###############################################################"""

##################################################
######################################## ADD SPEC and JIT
##################################################
@nb.experimental.jitclass()
class PhysicsUtilities:

    def __init__(self, D_x, v_kin, max_alpha=1, G=1, H=1):
        self.D_x = D_x
        self.v_kin = v_kin
        self.max_alpha = max_alpha
        self.G = G
        self.H = H

    def calculate_attachment_coefficient_with_kink(self, sat, kink):
        """
        
        """
        return self.max_alpha*np.exp(-1/(kink*sat))    

    def calculate_local_growth_velocity(self, sat, kink):
        """

        """
        alpha = self.calculate_attachment_coefficient_with_kink(sat, kink)

        return alpha*self.v_kin*sat

    def calculate_growth_time(self, sat, kink, f_b):
        """
        
        """
        v_growth = self.calculate_local_growth_velocity(sat, kink)
        growth_time_increment = self.H*self.D_x*(1-f_b)/v_growth

        return growth_time_increment

    def get_min_growth_time(filling_timing_array, boundary_cells):
        """
        
        """
        boundary_cell_amount = np.shape(boundary_cells)[0]
        growth_time_array = np.empty(boundary_cell_amount)
            
        for i in range(boundary_cell_amount):
            growth_time_array[i] = filling_timing_array[boundary_cells[i,:]]

        minimum_growth_time = np.min(growth_time_array)

        return minimum_growth_time

    ####################################################################
    ############################ FILL THIS OUT #########################
    ####################################################################
    def filling_factor_step(min_growth_time):
        """
        
        """


        return None

# def calculate_attachment_coefficient_with_kink(sat, kink, max_amplitude=1):
#     """
    
#     """
#     return max_amplitude*np.exp(-1/(kink*sat)) 


# def calculate_local_growth_velocity(sat, v_kin, kink):
#     """

#     """
#     alpha = calculate_attachment_coefficient_with_kink(sat, kink)

#     return alpha*v_kin*sat


# def calculate_growth_time(sat, v_kin, kink, f_b, D_x, H_b=1):
#     """
    
#     """
#     v_growth = calculate_local_growth_velocity(sat, v_kin, kink)
#     growth_time_increment = H_b*D_x*(1-f_b)/v_growth

#     return growth_time_increment


# def get_min_growth_time(filling_timing_array, boundary_cells):
#     """
    
#     """
#     boundary_cell_amount = np.shape(boundary_cells)[0]
#     growth_time_array = np.empty(boundary_cell_amount)
        
#     for i in range(boundary_cell_amount):
#         growth_time_array[i] = filling_timing_array[boundary_cells[i,:]]

#     minimum_growth_time = np.min(growth_time_array)

#     return minimum_growth_time


#def filling_factor_step(min_growth_time, H_b=1):
#     """
    
#     """


#     return None


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

    Return
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





# ###################### NOT SURE STILL NEEDED ##########################
# @nb.njit
# def calculate_sat_opp(sat_map, ice_map, local_neighbors):
#     """ Function that calculates the 
    
#     """
#     opposing_indices = np.array([3,4,5,0,1,2]) # in 'neighbor index' form
#     opp_cell_counter = 0
#     opp_cell_total = 0.0

#     for i in range(np.shape(local_neighbors)):
#         neighbor_line = local_neighbors[i,0]
#         neighbor_col = local_neighbors[i,1]

#         opp_line = local_neighbors[opposing_indices[i],0]
#         opp_col = local_neighbors[opposing_indices[i],1]

#         if ice_map[neighbor_line, neighbor_col] == True and ice_map[opp_line, opp_col] == False:
#             opp_cell_total += sat_map[opp_line, opp_col]
#             opp_cell_counter += 1

#     return opp_cell_total / opp_cell_counter # returns the average of the 'opposing' cells        
# #################################################################################

# def apply_boundary_condition():
#     # sigma = sigma_opp/(1+ alpha(sigma_old)G_b*Dx/X_0)

def get_opp_neighbor_indices(ice_map, local_neighbors):
    """A function that returns up to the three 
    
    """
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

    alpha = calculate_attachment_coefficient_with_kink(local_sat, kink_number)

    return sat_opp/(1 + alpha*G_b*D_x/X_0) # returns the new saturation of the boundary cell


def execute_relaxation_step(old_sat_map, normal_cells, boundary_cells, opp_array, neighbor_array, diffusion_rules, boundary_map, D_x):
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

    # application of boundary conditions to cells, I guess
    for i in range(boundary_cell_amount):
        line = int(boundary_cells[i,0])
        col = int(boundary_cells[i,1])

        local_neighbors = neighbor_array[line, col, :, :]
        local_diffusion_rules = diffusion_rules[line, col, :]
        local_opps = opp_array[i,:]
        
        new_sat_map[line, col] = apply_boundary_condition(
            line, 
            col, 
            old_sat_map, 
            local_neighbors, 
            local_opps, 
            boundary_map, 
            D_x
        )

    return new_sat_map


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


# FLUSH THIS FUNCTION OUT
def diffuse_to_convergence():
    # GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOD

    return None

# Think of growth steps

"""###############################################################
                        Growth utilities
###############################################################"""


"""
#### DO THING THAT INITIALIZES THE FILLING ARRAY AND THE TIME ARRAY
"""

def update_filling_factor():
    """
    
    """
    return None

######################################### NOT DONE
def update_time_array(time_array, boundary_cells):

    for coords in boundary_cells:
        delta_time = 0 ############################# MAKE THIS TING WORK
        time_array[coords[0], coords[1]] += delta_time
    

    return None


def update_filling_array(filling_array, boundary_cells):
    """
        REFERENCE OFF OF BOUNDARY CELLS 
    """

    for cell_coords in boundary_cells:
        pass ######################FIGURE THIS OUT

    return None ##########################FIGURE THIS OUT
