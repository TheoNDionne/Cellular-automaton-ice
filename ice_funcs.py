"""
Code used to simulate the formation of a C6v symmetric snowflakes.

!!! Re-implementation of bits of code found in numpy are due to !!!
!!! restricted numpy support in numba.                          !!!

Classes
-------
GeneralUtilities:
    A class that principally contains methods that allow to 
    generate the default neighbor finding array and calculate 
    the positions of boundary cells (with the amount of nearest
    neighbors).

PhysicsUtilities:
    A class that groups methods that execute "physical calculations", 
    such as the boundary growth velocity. Usually, methods there 
    depend on physical parameters.

SaturationRelaxationUtilities:
    A class that allows for two major things:
        - Relaxing the saturation field according to the diffusion 
          equation.
        - Applying the physical boundary conditions to the ice-vapor 
          boundary cells.

GrowthUtilities:
    A class that allows for the growth of the crystal by evolving the 'filling factor'
    of each boundary cell.

SnowflakeSimulation:
    A class that comprises the general parts of the simulation such 
    as the map of ice cells and the map of water saturation in the
    simulation.  


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
                    General utilities
###############################################################"""


@nb.experimental.jitclass({
    "L" : nb.int32,
    "W" : nb.int32,
    "neighbor_array" : nb.float32[:,:,:,::1]
})
class GeneralUtilities:
    """ Class that implements utilities that are related to general bookkeeping.
    """

    def __init__(self, L):
        """ Class initializer method.
        
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

    ### PRIVATE METHODS ###

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

    def _get_opp_neighbor_indices(self, ice_map, local_neighbors):
        """ A function that returns up to the three neighbor indices corresponding 
        to the three possible neighbor cells.
        """
        opposing_indices = np.array([3,4,5,0,1,2]) # in 'neighbor index' form
        local_opp_array = np.full(3, np.nan, dtype=np.float32)

        opps_detected = 0
        for i in range(np.shape(local_neighbors)[0]):
            neighbor_line = local_neighbors[i,0]
            neighbor_col = local_neighbors[i,1]

            opp_line = local_neighbors[opposing_indices[i],0]
            opp_col = local_neighbors[opposing_indices[i],1]

            if not np.isnan(neighbor_line) and not np.isnan(opp_line):
                if ice_map[int(neighbor_line), int(neighbor_col)] == True and ice_map[int(opp_line), int(opp_col)] == False:
                    local_opp_array[opps_detected] = opposing_indices[i] # adds opposing 'neighbor index'
                    opps_detected += 1

        return local_opp_array

    ### PUBLIC METHODS ###

    def distinguish_cells(self, ice_map, boundary_map):
        X_normal = []
        Y_normal = []
        
        X_boundary = []
        Y_boundary = []

        for line in range(1,self.L):
            for col in range((self.L-line+1)//2):
                if not ice_map[line, col]:
                    if boundary_map[line, col] == 0:
                        X_normal.append(line)
                        Y_normal.append(col)
                    else:
                        X_boundary.append(line)
                        Y_boundary.append(col)

        normal_cells = np.transpose(np.array([X_normal, Y_normal]))
        boundary_cells = np.transpose(np.array([X_boundary, Y_boundary]))
        
        return normal_cells, boundary_cells

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

                if not ice_map[line, col]:
                    for i in range(np.shape(neighbor_coords)[0]):
                        neighbor_line = neighbor_coords[i,0]
                        neighbor_col = neighbor_coords[i,1]

                        if not np.isnan(neighbor_line):
                            if ice_map[int(neighbor_line), int(neighbor_col)] == True:
                                neighbor_counter += 1

                boundary_map[line, col] = neighbor_counter

        return boundary_map
    
    def construct_opp_array(self, boundary_cells, ice_map, neighbor_array):
        """ Constructs an array that contains the coordinates of 'opposing cells'
        """
        array_length = np.shape(boundary_cells)[0]
        opp_array = np.empty((array_length, 3)) 
        
        for i in range(array_length):
            line = int(boundary_cells[i,0])
            col = int(boundary_cells[i,1])
            local_neighbors = neighbor_array[line, col, :, :]

            opp_array[i,:] = self._get_opp_neighbor_indices(ice_map, local_neighbors)

        return opp_array
                

# Define type of GeneralUtilities class instance
GeneralUtilities_instance_type = nb.deferred_type() # initialize GeneralUtilities instance type
GeneralUtilities_instance_type.define(GeneralUtilities.class_type.instance_type) # define GeneralUtilities' type as it's own type


"""###############################################################
                    Physics-based functions
         (mostly based on [arXiv:1910.06389, chap.5])
###############################################################"""


@nb.experimental.jitclass({
    "D_x" : nb.float32,
    "v_kin" : nb.float32,
    "max_alpha" : nb.float32,
    "b" : nb.float32,
    "X_0" : nb.float32,
    "G" : nb.float32,
    "H" : nb.float32,
    "safety_margin" : nb.float32
})
class PhysicsUtilities:

    def __init__(self, D_x, v_kin, b=1, max_alpha=1, X_0=1, G=1, H=1, safety_margin=1e-7):
        self.D_x = D_x
        self.v_kin = v_kin

        self.max_alpha = max_alpha
        self.b = b
        
        self.X_0 = X_0
        self.G = G
        self.H = H
        self.safety_margin = safety_margin


    def calculate_attachment_coefficient_with_kink(self, sat, kink):
        """ Method that determines the attachment coefficient as a function of local 
        water vapor saturation and the amount of nearest neighbor (kink number). 
        Loosely based off of [arXiv:1910.06389].
        
        """
        return self.max_alpha*np.exp(-self.b/(kink*sat+self.safety_margin))    


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


    def apply_boundary_condition(self, line, col, sat_map, sat_opp_avg, boundary_map):
        local_sat = sat_map[line, col]
        kink_number = boundary_map[line, col]

        alpha = self.calculate_attachment_coefficient_with_kink(local_sat, kink_number)

        return sat_opp_avg/(1 + alpha*self.G*self.D_x/self.X_0) # returns the new saturation of the boundary cell


    def calculate_filling_factor_increment(self, sat, kink, dt_min):
        """
        """
        v_growth = self.calculate_local_growth_velocity(sat, kink) 

        # inspired by K.G. Libbrecht
        return v_growth*dt_min/(self.H*self.D_x)


    def calculate_growth_time_increment(self, sat, kink, filling_factor):
        """
        """
        v_growth = self.calculate_local_growth_velocity(sat, kink)
        
        # inspired by K.G. Libbrecht
        return self.H*self.D_x*(1 - filling_factor)/(v_growth+self.safety_margin) 


# # Define type of PhysicsUtilities class instance
# PhysicsUtilities_instance_type = nb.deferred_type() # initialize PhysicsUtilities instance type
# PhysicsUtilities_instance_type.define(PhysicsUtilities.class_type.instance_type) # define PhysicsUtilities' type as it's own type


"""###############################################################
                    Diffusion utilities
###############################################################"""


@nb.experimental.jitclass({
    "L" : nb.int32,
    "W" : nb.int32,
    "max_iter" : nb.int32,
    "diffusion_rules" : nb.float32[:,:,::1]
})
class SaturationRelaxationUtilities:
    """FILL OUT"""

    def __init__(self, L, max_iter):
        """
        """
        self.L = L
        self.W = (L+1)//2

        self.max_iter = max_iter

        self.diffusion_rules = self._construct_default_diffusion_rules()


    ##### Private Methods #####


    def _construct_default_diffusion_rules(self):
        """ Constructs the 'vanilla' diffusion rules using the discretized
        version of the diffusion equation.

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
        inner_rule_set = np.ones(6, dtype=np.float32)/6
        
        #diffusion for the upper ragged cells
        upper_ragged_rule_set = np.array([
            1/6,
            1/6,
            0,
            1/6,
            1/6,
            1/3
        ], dtype=np.float32)

        #diffusion for the lower ragged cells
        lower_ragged_rule_set = np.array([
            1/3,
            0,
            0,
            0,
            1/3,
            1/3
        ], dtype=np.float32)

        # diffusion for the flush left cells
        flush_left_rule_set = np.array([
            1/6,
            1/3,
            1/3,
            1/6,
            0,
            0
        ], dtype=np.float32)
        
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


    def _diffuse_cell(self, sat_map, local_diffusion_rules, local_neighbors):
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
            line = local_neighbors[i,0]
            col = local_neighbors[i,1]
            if not np.isnan(line):
                new_sat += sat_map[int(line), int(col)]*local_diffusion_rules[i]
            
        return new_sat


    # Opp utilities #


    def _calculate_sat_opp_average(self, sat_map, local_neighbors, local_opps): ################################### WHISHY WASHY FUNCTION CALL ###########
        """ Calculates the average saturation of opposing cells located at the 
        
        """
        sat_sum = 0
        opp_counter = 0

        # loops over the 'opposing' cells
        for neighbor_index in local_opps:
            if not np.isnan(neighbor_index):
                opp_counter += 1 # increments the total count of opps
                opp_line = local_neighbors[int(neighbor_index), 0]
                opp_col = local_neighbors[int(neighbor_index), 1]

                sat_sum += sat_map[int(opp_line), int(opp_col)]

        if opp_counter == 0:
            return 0
        else:
            return sat_sum / opp_counter # returns the average of opposing saturations


    def _has_converged(self, sat_1, sat_2, epsilon):
        """ Function that allows for an element-wise convergence assessment
        of two arrays of the form |a-b| < epsilon. Only performs operation 
        over relevant physical sites.

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
        for line in range(1,self.L):
            for col in range((self.L-line+1)//2):
                if abs(sat_1[line,col] - sat_2[line,col]) > epsilon:
                    return False # returns False if any elements of the array are out of convergence

        return True # returns True if all elements of the array are withing convergence


    def _execute_relaxation_step(self, old_sat_map, normal_cells, boundary_cells, neighbor_array, boundary_map, opp_array, PU):
        """ JESUS CHRIST FIGURE THIS OUT LMAO.

        """
        new_sat_map = old_sat_map.copy() # initialize new sat map as a copy of the old sat

        # get amount of vapor cells of both types
        normal_cell_amount = np.shape(normal_cells)[0]
        boundary_cell_amount = np.shape(boundary_cells)[0]

        # diffuse all normal cells previously found
        for i in range(normal_cell_amount):
            line = int(normal_cells[i,0])
            col = int(normal_cells[i,1])

            local_neighbors = neighbor_array[line, col, :, :]
            local_diffusion_rules = self.diffusion_rules[line, col, :] # gets the local diffusion rules

            new_sat_map[line, col] = self._diffuse_cell(old_sat_map, local_diffusion_rules, local_neighbors)

        # application of boundary conditions to cells, I guess
        for i in range(boundary_cell_amount):
            line = int(boundary_cells[i,0])
            col = int(boundary_cells[i,1])

            local_neighbors = neighbor_array[line, col, :, :]
            local_diffusion_rules = self.diffusion_rules[line, col, :]
            local_opps = opp_array[i,:]

            sat_opp_avg = self._calculate_sat_opp_average(old_sat_map, local_neighbors, local_opps)
            
            new_sat_map[line, col] = PU.apply_boundary_condition(
                line, 
                col, 
                old_sat_map, 
                sat_opp_avg, 
                boundary_map
            )

        return new_sat_map


##### Public Methods #####


    def diffuse_to_convergence(self, sat_map, epsilon, normal_cells, boundary_cells, boundary_map, neighbor_array, opp_array, PU):
        """ Repeats the relaxation steps until the desired convergence is achieved.
        NOTE : epsilon is left as a dangly parameter because enveloppe class can adjust it.
        """
        # initializes `old_sat` as a copy of the given `sat_map`
        old_sat = sat_map.copy() 

        i = 0

        # tries the relaxation method for maximal amount of times
        while i < self.max_iter:
            i += 1 # increment loop counter

            # gets the new saturation map that's been relaxed one step
            new_sat = self._execute_relaxation_step(
                old_sat, 
                normal_cells, 
                boundary_cells,
                neighbor_array, 
                boundary_map,
                opp_array,
                PU
                )
            if self._has_converged(new_sat, old_sat, epsilon):
                return new_sat, True, i # returns reasonably converged saturation map and success status
            
            old_sat = new_sat.copy()

        return new_sat, False, i # returns unconverged array with failure status



############################################################################# ADD BACK WHEN DONE
# # Define type of SaturationRelaxationUtilities class instance
# SaturationRelaxationUtilities_instance_type = nb.deferred_type() # initialize SaturationRelaxationUtilities instance type
# SaturationRelaxationUtilities_instance_type.define(SaturationRelaxationUtilities.class_type.instance_type) # define SaturationRelaxationUtilities' type as it's own type


"""###############################################################
                        Growth utilities
###############################################################"""


"""
#### DO THING THAT INITIALIZES THE FILLING ARRAY AND THE TIME ARRAY
"""
@nb.experimental.jitclass({
    "L" : nb.int32,
    "W" : nb.int32,
    "filling_array" : nb.float32[:,::1],
    "minimum_time_increment" : nb.float32
})
class GrowthUtilities:
    ############################ FILL OUT COMMENTS #############################
    def __init__(self, L):
        self.L = L
        self.W = (L+1)//2
        self.filling_array = np.full((self.L, self.W), 0.0, dtype=np.float32) # initialize filling array

        self.minimum_time_increment = 0.0 # initialize minimum time increment

    ### PRIVATE METHODS ###

    ### PUBLIC METHODS ###

    # def update_timing_array(self, boundary_cells, sat_map, boundary_map):
    #     """
    #     ################# ONLY KEEP THE MINIMUM TIME INCREMENT!!
    #     """

    #     boundary_cell_amount = np.shape(boundary_cells)[0]
    #     time_increments = np.empty(boundary_cell_amount)

    #     for i in range(boundary_cell_amount):
    #         line = boundary_cells[i,0]
    #         col = boundary_cells[i,1]

    #         # calculate growth time increment
    #         growth_time_increment = self.PU.calculate_growth_time_increment(
    #             sat_map[line,col], 
    #             boundary_map[line,col], 
    #             self.filling_array[line,col]
    #         )
    #         time_increments[i] = growth_time_increment # add growth time increment to a temp array

    #         self.timing_array[line, col] += growth_time_increment  # add time increment to timing array

    #     # get minimum time increment and update minimum_time_increment attribute
    #     self.minimum_time_increment = np.min(time_increments) 

    #     return None


    def update_minimum_time(self, boundary_cells, sat_map, boundary_map, PU):
        """Calculates the minimum time increment in the active boundary cells
        """

        boundary_cell_amount = np.shape(boundary_cells)[0]
        time_increments = np.empty(boundary_cell_amount)

        for i in range(boundary_cell_amount):
            line = boundary_cells[i,0]
            col = boundary_cells[i,1]

            # calculate growth time increment
            growth_time_increment = PU.calculate_growth_time_increment(
                sat_map[line,col], 
                boundary_map[line,col], 
                self.filling_array[line,col]
            )
            time_increments[i] = growth_time_increment # add growth time increment to a temp array

        # get minimum time increment and update minimum_time_increment attribute
        self.minimum_time_increment = np.min(time_increments) 

        return None


    def update_filling_array(self, boundary_cells, sat_map, boundary_map, PU):
        """
        
        """

        for coords in boundary_cells: # only execute over boundary cells
            line = coords[0] # boundary cell line
            col = coords[1] # boundary cell col

            self.filling_array[line, col] += PU.calculate_filling_factor_increment(
                sat_map[line,col], 
                boundary_map[line, col], 
                self.minimum_time_increment
                )

        return None 


    # def get_cells_to_progress_from_filling_array(self):
    #     """Get cells that have a filling index > 1 and then set them to np.nan
    #     """
        
    #     filled_cell_lines = []
    #     filled_cell_cols = []

    #     for line in range(1,self.L):
    #         for col in range((self.L-line+1)//2):
    #             if self.filling_array[line,col] >= 1:
    #                 # collect filled cell coordinates
    #                 filled_cell_lines.append(line)
    #                 filled_cell_cols.append(col)

    #                 # set filled cells to np.nan to avoid recounting
    #                 self.filling_array[line,col] = np.nan

    #     return np.transpose(np.array([filled_cell_lines, filled_cell_cols]))


    def ammend_ice_map_from_filling(self, ice_map):
        """
        """
        # default value of the progression status
        ice_map_progressed = False

        for line in range(1,self.L):
            for col in range((self.L-line+1)//2):
                if self.filling_array[line,col] >= 1:
                    # update progression status to `True`
                    ice_map_progressed = True

                    # fill in appropriate cells in ice map
                    ice_map[line,col] = True

                    # set filled cells to np.nan to avoid recounting
                    self.filling_array[line,col] = np.nan

        return ice_map, ice_map_progressed

############################################################################# ADD BACK WHEN DONE
# # Define type of GrowthUtilities class instance
# GrowthUtilities_instance_type = nb.deferred_type() # initialize GrowthUtilities instance type
# GrowthUtilities_instance_type.define(GrowthUtilities.class_type.instance_type) # define GrowthUtilities' type as it's own type
    

"""###############################################################
                        Simulation class
###############################################################"""


class SnowflakeSimulation:

    def __init__(self, L, max_diffusion_iter=200, D_x=1, v_kin=1): 
        
        # simulation zone geometry
        self.L = L
        self.W = (L+1)//2

        # intialize global time
        self.global_time = 0.0

        # intialization of all useful subclasses
        self.GeneralU = GeneralUtilities(self.L)
        self.PhysicsU = PhysicsUtilities(D_x, v_kin) # make it so this can be changed!
        self.RelaxationU = SaturationRelaxationUtilities(self.L, max_diffusion_iter)
        self.GrowthU = GrowthUtilities(self.L)

    ### Private methods ###

    def _construct_hexagonal_ice_map(self, initial_seed_half_width):
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
        ice_map = np.full((self.L, self.W), False) # initialize 'empty' ice_map

        # set all cells within half breadth to ice
        for line in range(self.L-initial_seed_half_width, self.L):
            for col in range((self.L-line+1)//2):
                ice_map[line, col] = True

        # returns initialized ice map
        return ice_map


    ### Public methods ###


    def run_simulation(self, cycle_amount, epsilon=0.001, initial_seed_half_width=3, initial_sat=1): ################################# FILL OUT ARGUMENTS maybe move some
        
        """ STEP 0 : Initialize values """

        # initializing HEXAGONAL ice map seed
        ice_map = self._construct_hexagonal_ice_map(initial_seed_half_width)

        # intializing sat map 
        sat_map = np.full((self.L, self.W), initial_sat, dtype=np.float64)

        # Fetch the neighbor array
        neighbor_array = self.GeneralU.neighbor_array

        # Initializing quantities that must be generated every time ice map changes
        boundary_map = self.GeneralU.construct_boundary_map(ice_map)
        normal_cells, boundary_cells = self.GeneralU.distinguish_cells(ice_map, boundary_map)
        opp_array = self.GeneralU.construct_opp_array(boundary_cells, ice_map, neighbor_array)

        # status every 100 cycles
        for c in range(cycle_amount):
            if c%100 == 0:
                print(f"Iteration {c}")

            """ STEP I : Relax sat_map """

            # relaxes the sat map
            sat_map = self.RelaxationU.diffuse_to_convergence(
                sat_map, 
                epsilon, 
                normal_cells, 
                boundary_cells,
                boundary_map,
                neighbor_array,
                opp_array,
                self.PhysicsU
            )[0]

            """ STEP II : Update filling, timing and ice """
            # Update minimal time and then increment global time
            self.GrowthU.update_minimum_time(boundary_cells, sat_map, boundary_map, self.PhysicsU)
            self.global_time += self.GrowthU.minimum_time_increment

            self.GrowthU.update_filling_array(boundary_cells, sat_map, boundary_map, self.PhysicsU)
            ice_map, ice_map_progressed = self.GrowthU.ammend_ice_map_from_filling(ice_map)
            
        
            """ STEP III : Update boundary array *****and things like that***** """

            if ice_map_progressed:

                # reconstructing ice_map dependant quantities
                boundary_map = self.GeneralU.construct_boundary_map(ice_map)
                normal_cells, boundary_cells = self.GeneralU.distinguish_cells(ice_map, boundary_map)
                opp_array = self.GeneralU.construct_opp_array(boundary_cells, ice_map, neighbor_array)

            """ STEP IV : Assess if up to spec """
            
            ################################# Determine a good break condition

            
        return ice_map


if __name__ == "__main__":
    L = 1001

    gu = GeneralUtilities(L)
    ss = SnowflakeSimulation(L, 200)

    ice_map_test = ss._construct_minimal_ice_map()
    # ice_map_test[L-4, 0] = True 

    # sat_map_test = ss._initialize_sat_map()
    boundary_map_test = gu.construct_boundary_map(ice_map_test)

    print(ice_map_test[-5:,:5])
    # print(sat_map_test[-5:,:5])
    print(boundary_map_test[-5:,:5])
    