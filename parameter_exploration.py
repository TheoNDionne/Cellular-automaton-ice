"""Quick script to explore parameters. Writes results to specified file"""

# imports 
import numpy as np
import ice_funcs as icef
from csv import writer

""" THINGS TO BE CHANGED """

# simulation geometry
L = 551

# amount of cycles
CYCLES = 5000

# parameter values
B_VALUES = np.array([0.5,1,3,5,7,10,15])
MAX_ALPHA_VALUES = np.array([0.5,1,2])
D_X_VALUES = np.array([0.5,1,2])
X_0_VALUES = np.array([0.5,1,2])

# filename
OUTPUT_DIR = "exploration_first_pass"

""" PROCEDURE """

# initialize simulation/é
Simulation = icef.SnowflakeSimulation(L) 

# simulation info
simulation_counter = 0

# Open our existing CSV file in append mode
# Create a file object for this file
with open(f"{OUTPUT_DIR}/specs.csv", 'a') as f_object:

    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)

    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(["Simulation number", "b", "max_alpha", "D_x", "X_0"])

    # Close the file object
    f_object.close()

for b in B_VALUES:
    for max_alpha in MAX_ALPHA_VALUES:
        for D_x in D_X_VALUES:
            for X_0 in X_0_VALUES:
                #increment counter
                simulation_counter += 1

                # simulation parameters
                Simulation.PhysicsU.X_0 = 1
                Simulation.PhysicsU.max_alpha = 1
                Simulation.PhysicsU.b = 1
                Simulation.PhysicsU.D_x = 1

                # console update
                print(f"Simulating --- b={b} ; max_alpha={max_alpha} ; D_x={D_x} ; X_0={X_0}")

                ### actually simulating ###
                final_ice_map = Simulation.run_simulation(CYCLES)
                
                # params that we want to add as a new row
                List = [simulation_counter, b, max_alpha, D_x, X_0]
                
                # Open our existing CSV file in append mode
                # Create a file object for this file
                with open(f"{OUTPUT_DIR}/specs.csv", 'a') as f_object:
                
                    # Pass this file object to csv.writer()
                    # and get a writer object
                    writer_object = writer(f_object)
                
                    # Pass the list as an argument into
                    # the writerow()
                    writer_object.writerow(List)
                
                    # Close the file object
                    f_object.close()

                # saving ice map
                np.savetxt(f"{OUTPUT_DIR}/{simulation_counter}", final_ice_map)

