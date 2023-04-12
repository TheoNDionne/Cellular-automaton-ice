"""
Code used to simulate the formation of C6v symmetric snowflakes to 
via a 2D cellular automaton model.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! Re-implementation of bits of code found in numpy are due to !!!
!!! restricted numpy support in numba.                          !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
    depend on the physical constants and parameters.

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
    simulation. THIS CLASS ALLOWS FOR THE HIGH LEVEL SIMULATION OF 
    THE SNOWFLAKE.


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

    Attributes
    ----------
    L : int
        Number of cells in the long axis of the model
    W : int
        Number of cells in the short axis of the model
    neighbor_array : array(float, 4d)
        Array that keeps track of the neighbors. Array is indexed with the follo-
        wing indexes neighbor_array[i,j,k,l], where:
        - i : line of the cell for which to get neighbors.
        - j : column of the cell for which to get neighbors.
        - k : 'neighbor index' (6 possibilities for hexagonal cells).
        - l : toggle between line and column of neighbor at index `k`.

    Public Methods
    --------------
    distinguish_cells : 
        Generates an array of oordinates for normal gas cells and boundary gas cells.
    construct_boundary_array : 
        Constructs the boundary array that maps the kink number for every cell in the 
        simulation zone.
    construct_opp_array
        Constructs the opp array that contains up to 3 neighbor indices for every 
        boundary cell.
    """

    def __init__(self, L):
        """ Class initializer method.
        
        Attributes
        ----------
        L : int
            total length of the 1/12th slice of simulation zone.
        
        Return
        ------
        [None]
        """
        self.L = L
        self.W = (self.L+1)//2

        # 'One time runs' for geometry-dependent utilities
        self.neighbor_array = self._construct_neighbor_array() # construct neighbor array

    """PRIVATE METHODS"""

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
        Returns if the cell is a legit simulation cell. [bool]
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

        Return
        ------
        The neighbors of cell at position (line, col) in the model [array(float, 2d)].
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

        absolute_nearest_neighbors = np.empty((6, 2)) # intialize nearest neighbor sub-array

        for i in range(np.shape(relative_nearest_neighbors)[0]):
            neighbor_line = line + relative_nearest_neighbors[i,0] # absolute coordinate line of neighbor
            neighbor_col = col + relative_nearest_neighbors[i,1] # absolute coordinate col of neighbor

            if self._is_legit_cell(neighbor_line, neighbor_col):
                absolute_nearest_neighbors[i, :] = [neighbor_line, neighbor_col] # coordinates if valid neighbor
            else:
                absolute_nearest_neighbors[i, :] = [np.nan, np.nan] # empty value if not neighbor

        return absolute_nearest_neighbors

    def _construct_neighbor_array(self):
        """ Constructs the array of nearest neighbors for all cells in the model.

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
                # add in local neigbor array at every coordinate in model
                neighbor_array[line, col, :, :] = self._get_neighbors(line, col) 

        return neighbor_array

    def _get_opp_neighbor_indices(self, ice_map, local_neighbors):
        """ A function that returns up to the three neighbor indices corresponding 
        to the three possible opposing cells to a boundary cell.

        Arguments
        ---------
        ice_map : array(bool, 2d)
            Boolean mask representing ice status of a cell.
        local_neigbors : array(float, 2d)
            Restriction of neighbor array to neigbor_array[line,col,:,:]

        Return
        ------
        The local opposing cell array [array(float, 1d)]. If there's an opposing cell,
        then the `neighbor index` is present. If not, then element is `np.nan`.
        """
        opposing_indices = np.array([3,4,5,0,1,2]) # all potential neigbhor indices
        local_opp_array = np.full(3, np.nan, dtype=np.float32) # initialize local opp arr

        # zero opp counter
        opps_detected = 0
        for i in range(np.shape(local_neighbors)[0]): # check forall local neighbors
            # get coordinates of neighbor from neighbor index
            neighbor_line = local_neighbors[i,0]
            neighbor_col = local_neighbors[i,1]

            # get coordinates of potential opp cell from neighbor index
            opp_line = local_neighbors[opposing_indices[i],0]
            opp_col = local_neighbors[opposing_indices[i],1]

            if not np.isnan(neighbor_line) and not np.isnan(opp_line): # check that both coordinates exist (not np.nan)
                # if neigbhor is an ice cell and opp isn't, add neighbor index
                if ice_map[int(neighbor_line), int(neighbor_col)] == True and ice_map[int(opp_line), int(opp_col)] == False:
                    local_opp_array[opps_detected] = opposing_indices[i] # adds opposing 'neighbor index'
                    opps_detected += 1

        return local_opp_array

    """PUBLIC METHODS"""

    def distinguish_cells(self, ice_map, boundary_map):
        """ Function that creates two arrays of coordinates : one that contains the 
        boundary cells and one that contains the 'typical' gas cells.
        
        Attributes
        ----------
        ice_map : array(bool, 2d)
            Boolean mask representing ice status of a cell.
        boundary_map : array(int, 2d)
            Map that represents the amount of next-neighbor ice cells (kink) for 
            each cell.

        Return
        ------
        A tuple of numpy arrays representing the coordinates of normal gas cells 
        and boundary gas cells. [Tuple(array(int, 2d), array(int, 2d))]
        """
        #initialize coordinate lists
        X_normal = []
        Y_normal = []
        X_boundary = []
        Y_boundary = []

        # distinguishing over the possible simulation cells
        for line in range(1,self.L):
            for col in range((self.L-line+1)//2):
                if not ice_map[line, col]: # checks if not ice cell
                    if boundary_map[line, col] == 0: # appends to normal cell if kink=0
                        X_normal.append(line)
                        Y_normal.append(col)
                    else: # if kink=/=0 then append coords to boundary cell
                        X_boundary.append(line)
                        Y_boundary.append(col)

        # formatting both lists as numpy arrays
        normal_cells = np.transpose(np.array([X_normal, Y_normal]))
        boundary_cells = np.transpose(np.array([X_boundary, Y_boundary]))
        
        return normal_cells, boundary_cells

    def construct_boundary_map(self, ice_map): 
        """ Function that constructs a map of all boundary pixels. The number 
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
        The 2D map of the amount of neighboring ice cells (excluding ice cells 
        themselves). [array(int, 2d)]
        """
        boundary_map = np.zeros((self.L, self.W), dtype=np.int8) # initialize boundary map

        for line in range(self.L):
            for col in range((self.L - line + 1)//2):
                neighbor_coords = self.neighbor_array[line, col, :, :].copy() # returns 6x2 array
                
                neighbor_counter = 0 # zero amount of neighbors

                if not ice_map[line, col]: # check that cell isn't ice
                    for i in range(np.shape(neighbor_coords)[0]):
                        # neighbor coords from neighbor index
                        neighbor_line = neighbor_coords[i,0]
                        neighbor_col = neighbor_coords[i,1]

                        if not np.isnan(neighbor_line): # checks if neighbor exists in model
                            if ice_map[int(neighbor_line), int(neighbor_col)] == True:
                                neighbor_counter += 1 # increment number of neighbors if neigbor cell is ice

                # set the boundary_map to the amount of next-door neighbors
                boundary_map[line, col] = neighbor_counter

        return boundary_map
    
    def construct_opp_array(self, boundary_cells, ice_map, neighbor_array):
        """ Constructs an array that contains the coordinates of opposing cells for 
        every neighbor cell.

        Arguments
        ---------
        boundary_cells : array(int, 2d)
            List of coordinates of all the boundary cells in the model.
        ice_map : array(bool, 2d)
            The boolean map that represents the physical location of ice cells.
        neighbor_array : array(float, 4d)
            The array representing neigbors in the model of the form produced by 
            construct_neighbor_array().

            
        Return
        ------
        The array that contains the neighbor indices of a possible opposing cell.
        [array(float, 2d)]
        """
        boundary_cell_amount = np.shape(boundary_cells)[0]
        opp_array = np.empty((boundary_cell_amount, 3)) # initialize opp array
        
        for i in range(boundary_cell_amount): # append an element for every boundary cell
            line = int(boundary_cells[i,0])
            col = int(boundary_cells[i,1])
            local_neighbors = neighbor_array[line, col, :, :] # getting local neighbor array for boundary coord

            opp_array[i,:] = self._get_opp_neighbor_indices(ice_map, local_neighbors) # update to opp array

        return opp_array
                

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
    """ Class that incorporates methods that implement physical processes as 
    described in 'Snow Crystals' by K.G. Libbrecht.
    NOTE : All operation that depend on physical constants are located here!

    Attributes
    ----------
    D_x : float
        Physical size of a cell as measured via the the distance between 
        two opposing summits of a hexagonal cell.
    v_kin : float
        Physical velocity of water vapor particles in air.
    b : float
        Parameter with the dimensions of the water supersaturation field. 
        Regulates the shape of the exponential.
    max_alpha : float
        Parameter with the units of the attachment coefficient. Regulates 
        the maximum value of the attachment coefficient.
    X_0 : float
        Characteristic diffusion length in the model.
    G : float
        Intrinsic unitless geometrical correction factor for the attachment process.
    H : float
        Intrinsic unitless geometrical correction factor for the growth process.
    safety_margin : float
        Small float for zero division protection

    Public methods
    ---------------
    calculate_attachment_coefficient_with_kink : 
        Function that represents the physical attachment coefficient functionnal.
    
    """

    def __init__(self, D_x=1, v_kin=1, b=1, max_alpha=1, X_0=1, G=1, H=1, safety_margin=1e-7):
        """Class initializer method"""

        # simulation parameters
        self.D_x = D_x
        self.X_0 = X_0
        self.v_kin = v_kin

        # attachment coefficient
        self.max_alpha = max_alpha
        self.b = b
        
        # optionnal geometrical factors
        self.G = G
        self.H = H

        # safety margin to keep division by zero
        self.safety_margin = safety_margin

    ### PRIVATE METHODS ###

    def calculate_attachment_coefficient_with_kink(self, sat, kink):
        """ Method that determines the attachment coefficient as a function of local 
        water vapor saturation and the amount of nearest neighbor (kink number). 
        Loosely based off of [arXiv:1910.06389].

        Arguments
        ---------
        sat : float
            Value of the supersaturation field locally.
        kink : int
            Amount of nearest neighbors.

        Return
        ------
        Attachment coefficient. [float]
        """
        return self.max_alpha*kink*np.exp(-self.b/(kink*sat+self.safety_margin))    


    def calculate_local_growth_velocity(self, sat, kink):
        """ Calculates the growth velocity of the ice surface locally.

        Arguments
        ---------
        sat : float
            Value of the supersaturation field locally.
        kink : int
            Amount of nearest neighbors.

        Return
        ------
        Local growth velocity coefficient. [float]
        """
        alpha = self.calculate_attachment_coefficient_with_kink(sat, kink)

        return alpha*self.v_kin*sat


    def calculate_growth_time(self, sat, kink, filling_factor):
        """ Calculates time increment required for a given cell to "grow" 
        as a function of its filling factor.

        Arguments
        ---------
        sat : float
            Value of the supersaturation field locally.
        kink : int
            Amount of nearest neighbors.
        filling_factor : float
            A real number between 0 and 1 representing how close a given 
            boundary cell is to becoming ice.
        
        Return
        ------
        The growth velocity at a given boundary cell. [float]
        """
        v_growth = self.calculate_local_growth_velocity(sat, kink)
        growth_time_increment = self.H*self.D_x*(1-filling_factor)/v_growth

        return growth_time_increment


    def get_min_growth_time(filling_timing_array, boundary_cells):
        """ Function that calculates ALL growth times and then returns the shortest one.

        Arguments
        ---------
        filling_timing_array : array(float, 2d)
            An array containing all the times the growth times for the given iteration 
            of the growth step (indexed by position).
        boundary_cells : array(int, 2d)
            List of coordinates of all the boundary cells in the model.
            
        Return
        ------
        The shortest growth time. [float]
        """
        boundary_cell_amount = np.shape(boundary_cells)[0]
        growth_time_array = np.empty(boundary_cell_amount)
            
        for i in range(boundary_cell_amount):
            growth_time_array[i] = filling_timing_array[boundary_cells[i,:]]

        minimum_growth_time = np.min(growth_time_array)

        return minimum_growth_time


    def apply_boundary_condition(self, line, col, sat_map, sat_opp_avg, boundary_map):
        """Function that applies the vapor-ice boundary condition by returning a saturation 
        value as a function of local and near-neighbor values.
        
        Arguments
        ---------
        line : int
            Line of the cell at which to apply boundary_condition.
        col : int
            Column at which to apply boundary condition.
        sat_map : array(float, 2d)
            The saturation field.
        sat_opp_avg : 
            Average value of the opposing cells of the given boundary cell.
        boundary_map : array(int, 2d)
            Map that represents the amount of next-neighbor ice cells (kink) for 
            each cell.

        Return
        ------
        The new saturation on the boundary pixel as defined by the boundary condition.
        """
        local_sat = sat_map[line, col] # get local saturation
        kink_number = boundary_map[line, col] # get amount of local neighbors

        # calculate attachment coefficient
        alpha = self.calculate_attachment_coefficient_with_kink(local_sat, kink_number)

        return sat_opp_avg/(1 + alpha*self.G*self.D_x/self.X_0) # returns the new saturation of the boundary cell


    def calculate_filling_factor_increment(self, sat, kink, dt_min):
        """ Calculate by how much to increment the filling factor of each boundary cell.

        Arguments
        ---------
        sat : float
            Value of the supersaturation field locally.
        kink : int
            Amount of nearest neighbors.
        dt_min : float
            Value of the smallest growth time interval in the current step of the 
            growth cycle.

        Return
        ------
        Number by which to increment the filling factor. [float]
        """
        v_growth = self.calculate_local_growth_velocity(sat, kink) # growth velocity

        # inspired by K.G. Libbrecht
        return v_growth*dt_min/(self.H*self.D_x)


    def calculate_growth_time_increment(self, sat, kink, filling_factor):
        """ Calculate by how much to increment the growth time of each boundary cell.

        Arguments
        ---------
        sat : float
            Value of the supersaturation field locally.
        kink : int
            Amount of nearest neighbors.
        filling_factor : float
            Filling factor of the cell for which the growth time increment must be 
            calculated.

        Return
        ------
        The number by which to increment the growth time of the specified boundary cell.
        [float]
        """
        v_growth = self.calculate_local_growth_velocity(sat, kink)
        
        # inspired by K.G. Libbrecht
        return self.H*self.D_x*(1 - filling_factor)/(v_growth+self.safety_margin) 


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
    """ A class that regroups methods with the common purpose of allowing for the 
    
    """

    def __init__(self, L, max_iter):
        """ Class contructor method.
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


"""###############################################################
                        Growth utilities
###############################################################"""


@nb.experimental.jitclass({
    "L" : nb.int32,
    "W" : nb.int32,
    "filling_array" : nb.float32[:,::1],
    "minimum_time_increment" : nb.float32
})
class GrowthUtilities:
    """
    FDAFADFAFASFas
    """
    def __init__(self, L):
        self.L = L
        self.W = (L+1)//2
        self.filling_array = np.full((self.L, self.W), 0.0, dtype=np.float32) # initialize filling array

        self.minimum_time_increment = 0.0 # initialize minimum time increment

    ### PUBLIC METHODS ###

    def update_minimum_time(self, boundary_cells, sat_map, boundary_map, PU):
        """Calculates the minimum time increment in the active boundary cells 
        and updates the minimum_time_increment_attricute
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


"""###############################################################
                        Simulation class
###############################################################"""


class SnowflakeSimulation:
    """ A class that handles the actual simulating of the snowflakes

        Attributes
        ----------
        L : int
            Number of cells in the long axis of the model
        W : int
            Number of cells in the short axis of the model
        global_time : float
            The total elapsed time of the simulation
        GeneralU : class instance object
            An instance of GeneralUtilities
        PhysicsU : class instance object
            An instance of PhysicsUtilities
        RelaxationU : class instance object
            An instance of RelaxationUtilities
        GrowthU : class instance object
            An instance of GrowthUtilities
            
        Public methods
        --------------
        run_simulation : 
            the method that simulates a snowflake and returns the 
            appropriate ice_array
        """
    
    def __init__(self, L, max_diffusion_iter=200, D_x=1, v_kin=1): 
        """ Class initializer method, nuff said
        """

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

    def run_simulation(self, cycle_amount, epsilon=0.001, initial_seed_half_width=3, initial_sat=1):
        """ The principal function used for simulating the snowflakes

        Arguments
        ---------
        cycle_amount : int
            Number of cycles in the cellular automaton algorithm
        epsilon : float
            Tolerance at which the supersaturation is considered converged
        initial_seed_half_width : int
            Half width of the initial ice crystal counted in cells
        initial_sat : float
            Initial fill value of the saturation map
        
        Return
        ------
        The ice map of the generated crystal (array(bool, 2d))
        
        """


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

            # Update the filling array
            self.GrowthU.update_filling_array(boundary_cells, sat_map, boundary_map, self.PhysicsU)
            ice_map, ice_map_progressed = self.GrowthU.ammend_ice_map_from_filling(ice_map)
            
        
            """ STEP III : Update boundary array *****and things like that***** """

            if ice_map_progressed:
                # reconstructing ice_map dependant quantities
                boundary_map = self.GeneralU.construct_boundary_map(ice_map)
                normal_cells, boundary_cells = self.GeneralU.distinguish_cells(ice_map, boundary_map)
                opp_array = self.GeneralU.construct_opp_array(boundary_cells, ice_map, neighbor_array)

        return ice_map


if __name__ == "__main__":
    """
    Does a test to check if simulation is running correctly by comparing 
    a pre-generated ice map with a freshly generated ice map.
    """
    
    L = 551
    CYCLES = 1000

    Simulation = SnowflakeSimulation(L) # init of simulation object
    ice_map = Simulation.run_simulation(CYCLES) # running default simulation

    reference_ice_map = np.genfromtxt("data/reference_ice_map.csv", dtype=np.int8).astype("bool") # fetch reference ice array

    # compare reference with default to confirm if simulation is working
    if np.array_equal(ice_map, reference_ice_map):
        print("Test passed")
    else:
        print("Test failed")