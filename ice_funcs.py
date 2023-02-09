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
 \_____/   X   \_____/       ======>      |   5   |   X   |   2   |
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
                    General utilities
###############################################################"""


@nb.experimental.jitclass({
    "L" : nb.int32,
    "W" : nb.int32,
    "neighbor_array" : nb.float32[:,:,:,::1],
    "diffusion_rules" : nb.float32[:,:,::1]
})
class GeneralUtilities:
    """ Class that implements utilities that are related to general bookkeeping.
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

        # 'One time runs' for geometry-dependent utilities
        self.neighbor_array = self._construct_neighbor_array() # construct neighbor array
        self.diffusion_rules = self._construct_default_diffusion_rules() # construct diffusion rules


    def _is_legit_cell(self, line, col):
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

    def _get_neighbors(self, line, col):
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

            if self._is_legit_cell(neighbor_line, neighbor_col):
                absolute_nearest_neighbors[i, :] = [neighbor_line, neighbor_col]
            else:
                absolute_nearest_neighbors[i, :] = [np.nan, np.nan]

        return absolute_nearest_neighbors

    def _construct_neighbor_array(self):
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
        neighbor_array = np.empty((self.L, self.W, 6, 2), dtype=np.float32) # float type to allow np.nan

        for line in range(self.L):
            for col in range((self.L - line + 1)//2):
                neighbor_array[line, col, :, :] = self._get_neighbors(line, col)

        return neighbor_array

    def _construct_default_diffusion_rules(self):
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
        
        diffusion_rules = np.empty((self.L, self.W, 6), dtype=np.float32)

        # fill out cells pertaining to flush left cells
        for line in range(1, self.L-4):
            diffusion_rules[line, 0, :] = flush_left_rule_set

        # fill out all lower ragged cells
        for col in range(2, self.W-1):
            line = self.L - 2*col - 1
            diffusion_rules[line, col, :] = lower_ragged_rule_set

        # fill out all upper ragged cells
        for col in range(2, self.W-1):
            line = self.L - 2*col - 2
            diffusion_rules[line, col, :] = upper_ragged_rule_set

        # fill out the rest
        for col in range(1, self.W-1):
            for line in range(1, self.L - 2*col - 2):
                diffusion_rules[line, col, :] = inner_rule_set

        return diffusion_rules

    def construct_boundary_map(self, ice_map): 
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
        boundary_map = np.zeros((self.L, self.W), dtype=np.int8) # initialize boundary map

        for line in range(self.L):
            for col in range((self.L - line + 1)//2):
                neighbor_coords = self.neighbor_array[line, col, :, :].copy() # returns 6x2 array
                
                neighbor_counter = 0

                if ice_map[line, col] != True:
                    for i in range(np.shape(neighbor_coords)[0]):
                        neighbor_line = int(neighbor_coords[i,0]) # converts line to integer value
                        neighbor_col = int(neighbor_coords[i,1]) # converts col to integer value

                        if not np.isnan(neighbor_line):
                            if ice_map[neighbor_line, neighbor_col] == True:
                                neighbor_counter += 1

                boundary_map[line, col] = neighbor_counter

        return boundary_map



"""###############################################################
                    Physics-based functions
            (all based on [arXiv:1910.06389, chap.5])
###############################################################"""


@nb.experimental.jitclass({
    "D_x" : nb.float32,
    "v_kin" : nb.float32,
    "max_alpha" : nb.float32,
    "X_0" : nb.float32,
    "G" : nb.float32,
    "H" : nb.float32
})
class PhysicsUtilities:

    def __init__(self, D_x, v_kin, max_alpha=1, X_0=1, G=1, H=1):
        self.D_x = D_x
        self.v_kin = v_kin
        self.max_alpha = max_alpha
        self.X_0 = X_0
        self.G = G
        self.H = H

    def calculate_attachment_coefficient_with_kink(self, sat, kink):
        """ Method that determines the attachment coefficient as a function of local 
        water vapor saturation and the amount of nearest neighbor (kink number). 
        Loosely based off of [arXiv:1910.06389].
        
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

    def filling_factor_step(min_growth_time):
        """
        
        """
        ####################################################################
        ############################ FILL THIS OUT #########################
        ####################################################################

        return None

    def apply_boundary_condition(self, line, col, sat_map, local_neighbors, local_opps, boundary_map):
        local_sat = sat_map[line, col]
        kink_number = boundary_map[line, col]
        sat_opp = calculate_sat_opp_average(sat_map, local_neighbors, local_opps)

        alpha = self.calculate_attachment_coefficient_with_kink(local_sat, kink_number)

        return sat_opp/(1 + alpha*self.G_b*self.D_x/self.X_0) # returns the new saturation of the boundary cell

# Define the type of an instance of PhysicsUtilities class to be passed as an argument to constructor of other classes
PhysicsUtilities_instance_type = nb.deferred_type() # initialize PhysicsUtilities instance type
PhysicsUtilities_instance_type.define(PhysicsUtilities.class_type.instance_type) # define PhysicsUtilities' type as it's own type

"""###############################################################
                    Diffusion utilities
###############################################################"""


#####################################################
############# RESTRUCTURE TO MAKE FULLY OO ##########
#####################################################
class SaturationRelaxationUtilities:
    """FILL OUT"""

    def __init__(self):
        # FILL
        A = 1

    def diffuse_cell(self, sat_map, local_diffusion_rules, local_neighbors):
        """ Returns the value of a cell after one step of the diffusion process.

        Arguments
        ---------
        sat_map : array(float, 2d)
            The saturation field before the diffusion step.
        local_diffusion_rules : array(float, 1d)
            The local diffusion rules as a slice of diffusion_rules[i,j,:].
        local_neighbors : array(float, 2d)
            The local neighbors as a slice of neighbor_array[i,j,:,:].

        Return
        ------
        The new sat map after a diffusion iteration. (array(float, 2d))
        """
        new_sat = 0 # initialize new saturation 
        
        for i in range(np.shape(local_neighbors)[0]):
            line = int(local_neighbors[i,0])
            col = int(local_neighbors[i,1])

            new_sat += sat_map[line, col]*local_diffusion_rules[i]
            
        return new_sat

    def get_opp_neighbor_indices(self, ice_map, local_neighbors):
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

    def distinguish_cells(self, ice_map, boundary_map, l):
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

    def construct_opp_array(self, boundary_cells, ice_map, neighbor_array):
        array_length = np.shape(boundary_cells)[0]
        opp_array = np.empty((array_length, 3)) 
        
        for i in range(array_length):
            line = int(boundary_cells[i,0])
            col = int(boundary_cells[i,1])
            local_neighbors = neighbor_array[line, col, :, :]

            #############################################################################
            ########################### FIX THIS ########################################
            #############################################################################
            # opp_array[i,:] = get_opp_neighbor_indices(ice_map, local_neighbors)

        return opp_array

    def calculate_sat_opp_average(self, sat_map, local_neighbors, local_opps):
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

    def execute_relaxation_step(self, old_sat_map, normal_cells, boundary_cells, opp_array, neighbor_array, diffusion_rules, boundary_map, D_x):
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

            new_sat_map[line, col] = self.diffuse_cell(old_sat_map, local_diffusion_rules, local_neighbors)

        # application of boundary conditions to cells, I guess
        for i in range(boundary_cell_amount):
            line = int(boundary_cells[i,0])
            col = int(boundary_cells[i,1])

            local_neighbors = neighbor_array[line, col, :, :]
            local_diffusion_rules = diffusion_rules[line, col, :]
            local_opps = opp_array[i,:]
            
            ######################################################################
            ########################## FIGURE THIS OUT ###########################
            ###################################################################### 
            # new_sat_map[line, col] = apply_boundary_condition(
            #     line, 
            #     col, 
            #     old_sat_map, 
            #     local_neighbors, 
            #     local_opps, 
            #     boundary_map, 
            #     D_x
            # )
            ######################################################################

        return new_sat_map

    def has_converged(self, sat_1, sat_2, epsilon):
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

    def diffuse_to_convergence(self):
        # GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOD

        return None



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



if __name__ == "__main__":
    L = 1001

    gu = GeneralUtilities(L)

    ice_map = construct_minimal_ice_map(L, (L+1)//2)
    print(ice_map[-7:, :7])
    print(gu.construct_boundary_map(ice_map)[-5:,:5])
    print(gu.construct_boundary_map(ice_map)[-5:,:5])


    # bm_2 = gu.construct_boundary_map(ice_map)

    # print(bm_2[-5:, :5])

    # print(nb.typeof(gu.neighbor_array))
    # # print(nb.typeof(gu.diffusion_rules))
    # print(nb.typeof(gu.neighbor_array))