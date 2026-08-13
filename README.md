To reproduce the figures in the preprint:<br>
Separation capacity of linear reservoirs with random connectivity matrix
https://arxiv.org/abs/2404.17429

<h2> Figures :</h2>

Run **main_figures.py**.
You can safely comment the corresponding parts of the program that are of no interest.

**Figures 1 and 4:**<br>
Run the corresponding part with the parameters:
- T =  length of time series (e.g. 12),
- N = maximal reservoir dimension (e.g. 100),
- MC_steps = Number of Monte Carlo simulations to generate the matrix of moments (e.g. 100000),
- sym = "sym" or "iid" (the symmetry condition on the random connectivity matrix).
- List of standard deviations: alpha_list = the list of exponents to be considered in the scalings (e.g. [0.25, 0.5, 0.85, 1.00], corresponding to
                standard deviations [1/N^{0.25}, 1/N^{0.5}, ...])


**Figure 2:**<br>
Run the corresponding part with the parameters:
- T =  length of time series (e.g. 12),
- rho_list = the list of the std deviations (e.g [0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2])


**Figures 3 and 6:**<br>
Run the corresponding part with the parameters:
- T =  maximal length of time series (e.g. 12)
- N = reservoir dimension (e.g. 100)
- MC_steps = Number of Monte Carlo simulations to generate the matrix of moments (e.g. 100000)
- sym = "sym" or "iid" (the symmetry condition on the random connectivity matrix)


**Figure 5:**<br>
Run the corresponding part with the parameters:
- T = length of time series (e.g. 12)
- rho_list = the list of the std deviations (e.g [0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2]).

<h2> Tables :</h2>

**Tables 1 and 2:**<br>
Run **ecg5000_classification.py**.
Choose the parameters:
- N = reservoir dimension (e.g. 50),
- p = number of simulations of the random connectivity matrix (e.g. 50),
- alpha = exponent for the connectivity entry standard deviation = 1/N^alpha (e.g. 0.6), 
- N_EPOCHS = number of training epochs (e.g. 300), 
- TRAINABLE  = whether to train the input mask u (option "no" or "yes").

**Tables 3 and 4:**<br>
Run **pi_digits_memory.py**.
Choose the parameters:
- N = reservoir dimension (e.g. 50),
- T = length of/number of digits in the pi expansion (e.g. 300),
- d = delay: number of steps in the past that the neural network needs to remember (e.g. 20),
- p = number of simulations of the random connectivity matrix (e.g. 50),
- alpha = exponent for the connectivity entry standard deviation = 1/N^alpha (e.g. 0.6), 
- N_EPOCHS = number of training epochs (e.g. 300), 
- TRAINABLE  = whether to train the input mask u (option "no" or "yes").


**Tables 5 and 6:**<br>
Run **lorenz_forecasting.py**.
Choose the parameters:
- N = reservoir dimension (e.g. 50),
- T = number of steps in the Lorenz system simulation (e.g. 300),
- p = number of simulations of the random connectivity matrix (e.g. 50),
- alpha = exponent for the connectivity entry standard deviation = 1/N^alpha (e.g. 0.6),
- TRAINABLE  = whether to train the input mask u (option "no" or "yes").