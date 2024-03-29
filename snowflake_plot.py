""" snowflake_plot.py

Implementation of a class useful for plotting snowflakes C6v 
symmetric snowflakes using the 'ice_map' array. The crucial 
method is `CoordsFromIceMap.convert_ice_map_to_all_coords()`.
"""

import numpy as np
import numba as nb
from math import sqrt, cos, sin, pi


# specifications of datatypes in class
spec = {
    "L" : nb.int32,
    "W" : nb.int32
}
@nb.experimental.jitclass(spec)
class CoordsFromIceMap:
    """ A class that allows for the conversion of a boolean `ice_map` array
    from `ice_funcs.py` to coordinates to be used by matplotlib or an equi-
    valent module.
    
    Attributes
    ----------
    L : int
        Number of cells in the long axis of the model
    W : int
        Number of cells in the short axis of the model

    Public Methods
    --------------
    convert_ice_map_to_all_coords :
        method that takes ice_map and converts them to plot-ready coordinates.
    """
    def __init__(self, L):
        self.L = L
        self.W = (L+1)//2

    def __repr__(self):
        length = str(self.L)
        width = str(self.W)

        # returns length and width of simulation zone
        return "length = " + length + " ; " + "width = " + width

    def _coords_from_array_position(self, line, col):
        """ Method that converts array positions (line, col) to 
        pseudo-cartesian coordinates with an origin at cell (L-1,0).

        Arguments
        ---------
        line : int
            The line of the ice_cell in ice_map.
        col : int
            The column of the ice_cell in ice_map.

        Return
        ------
        (tuple) Returns the x coordinate and then the y coordinate of
        specified cell as (x,y).
        """
        # translate (line, col) to (x,y) with origin at (line, col) = (L-1,0)
        array_x = col
        array_y = self.L - line - 1

        # make cartesian coordinates
        x_coord = sqrt(3)*array_x/2
        y_coord = array_y - 0.5*array_x

        return x_coord, y_coord # coordinate tuple

    def _construct_coords_from_ice_map(self, ice_map):
        """ Function that reads over a boolean 'ice map' and creates 
        a pair of coordinates for every ice cell and returns coordinates
        as an array of coordinates. Automatically applies the reflexion 
        boundary condition that exists along the edge defined by col=0.

        Arguments
        ---------
        ice_map : array(bool, 2d)
            Boolean map of ice cells (True if ice, False or else).

        Return
        ------
        (array(float, shape=(:,2)))
        """
        # initialize coordinate lists
        x_coords = []
        y_coords = []

        # translate and add coords from col=0
        for line in range(self.L):
            if ice_map[line, 0] == True: # checks if ice cell
                x, y = self._coords_from_array_position(line, 0) 

                x_coords.append(x)
                y_coords.append(y)

        # translate and add all other ice cell coords and their reflexion about x=0
        for col in range(1, self.W):
            for line in range(self.L-2*col):
                if ice_map[line, col] == True: # checks if ice cell
                    x, y = self._coords_from_array_position(line, col) 

                    # append cell with positive x
                    x_coords.append(x)
                    y_coords.append(y)

                    # append cell with negative x (reflexion)
                    x_coords.append(-x)
                    y_coords.append(y)

        return np.transpose(np.array([x_coords, y_coords])) # returns array of ice cell coordinates

    def _rotate_coordinate_array(self, vectors, angle):
        """ A function that takes an array of cartesian coordinate 
        vectors and rotates them by the specified polar angle (in radians).

        Arguments
        ---------
        vectors : array(float, 2d)
            Array of cartesian vector coordinates in the shape produced by 
            _construct_coords_from_ice_map().
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

    def _reconstruct_all_coordinates(self, vectors):
        """ A function that takes all the vector coordinates of the ice cells 
        in the upper sixth of the hexagonal mesh (i.e. the reflected simulation 
        zone) and rotates it to fill entire hexagonal grid.

        Arguments
        ---------
        vectors : array(float, 2d)
            Array of cartesian vector coordinates in the shape produced by 
            _construct_coords_from_ice_map().

        Return
        ------
        (array(float, 2d)) Array of cartesian vector coordinates for all cells 
        in the shape produced by _construct_coords_from_ice_map().   
        """
        # generates list of rotation angles in radians (1/6th of a turn 5 times)
        angles = 2*pi*np.arange(60, 361, 60)/360 
        number_of_vectors = np.shape(vectors)[0] # amount of vectors in the upper sixth of the zone
        
        all_coordinates = np.empty((number_of_vectors*6, 2)) # initialize total position vector array
        all_coordinates[:number_of_vectors, :] = vectors # initialize first sixth of vector array

        # Add rotated cell coords to total coords array
        for i in range(5):
            all_coordinates[(i+1)*number_of_vectors:(i+2)*number_of_vectors, :] = self._rotate_coordinate_array(vectors, angles[i])

        return all_coordinates

    def convert_ice_map_to_all_coords(self, ice_map):
        """ A wrapper function that converts the provided ice map 
        to an array of cartesian coordinates representing the center 
        positions of all cells.

        Arguments
        ---------
        ice_map : array(bool, 2d)
            Boolean map of ice cells (True if ice, False or else).

        Return
        ------
        (array(float, 2d)) Array of cartesian vector coordinates for all cells 
        in the shape produced by _construct_coords_from_ice_map().   
        """
        upper_sixth_positions = self._construct_coords_from_ice_map(ice_map) # generate positions for the upper sixth
        all_positions = self._reconstruct_all_coordinates(upper_sixth_positions) # turn sixth into entire coordinates for all cells

        return all_positions