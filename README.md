# Cellular-automaton-ice

This repo contains the code used to generate 2 dimensionnal C6v symmetric snowflakes using a cellular automaton model based on methods descibed in *Snow Crystals* by K.G. Libbrecht at Caltech. The main files comprising useful code are:

- `ice_funcs.py` : the module that contains the code to perform the cellular automaton simulation
- `snowflake_plot.py` : a module containing code that translates the 2D boolean array representing the locations of ice cells to cartesian coordinates of the cells representing the entire snowflake.

Also the following files are useful for exploring the capabilities of the code:

- `graphiques_resultats.ipynb` : Plots snowflakes used for the report. Data was generated using `parameter_exploration.py`
- `snowflake_playground.ipynb` : A notebook presenting the generation and plotting of a snowflake using both modules described above.

This project was done with the goal of being the final assignment of *PHQ404 : méthodes numériques et simulations* at the *Université de Sherbrooke* over the winter 2023 session.