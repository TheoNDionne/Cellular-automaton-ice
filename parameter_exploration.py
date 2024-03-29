"""Quick script to explore parameters. Writes results to specified file"""

# imports 
import numpy as np
import ice_funcs as icef
from csv import writer

""" THINGS TO BE CHANGED """

# simulation geometry
L = 551

# amount of cycles
CYCLES = 20000

# parameter values
B_VALUES = np.array([0.05, 0.1, 0.2,0.5,1,2,5,10])
MAX_ALPHA_VALUES = np.array([1])
D_X_VALUES = np.array([1])
X_0_VALUES = np.array([1])

# filename
OUTPUT_DIR = "exemple_raw_sweep"

""" PROCEDURE """ 

# simulation info
simulation_counter = 0

# Open our existing CSV file in append mode
# Create a file object for this file
with open(f"{OUTPUT_DIR}/specs.csv", 'w+') as f_object:

    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)

    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(["Simulation number", "b", "max_alpha", "D_x", "X_0", "global_time", f"Cycles={CYCLES}", f"L={L}"])

    # Close the file object
    f_object.close()

for b in B_VALUES:
    for max_alpha in MAX_ALPHA_VALUES:
        for D_x in D_X_VALUES:
            for X_0 in X_0_VALUES:
                # initialize simulation
                Simulation = icef.SnowflakeSimulation(L) # I have no clue why this wasn't working

                #increment counter
                simulation_counter += 1

                # simulation parameters
                Simulation.PhysicsU.X_0 = X_0
                Simulation.PhysicsU.max_alpha = max_alpha
                Simulation.PhysicsU.b = b
                Simulation.PhysicsU.D_x = D_x

                # console update
                print(f"Simulating --- b={b} ; max_alpha={max_alpha} ; D_x={D_x} ; X_0={X_0}")

                ### actually simulating ###
                final_ice_map = Simulation.run_simulation(CYCLES)
                


                # params that we want to add as a new row
                List = [simulation_counter, b, max_alpha, D_x, X_0, Simulation.global_time]
                
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

