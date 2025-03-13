To reproduce the figures in the preprint:
Separation capacity of linear reservoirs with random connectivity matrix
https://arxiv.org/abs/2404.17429

Run main_figures.py.
You can safely comment the corresponding parts of the program that are of no interest.

Figure 1:
Run the corresponding part with the parameters:
T = 10 (length of time series)
alpha = standard deviation (e.g. 0.25, 1 or 1.5 in the paper)
distro = "Gauss"


Figure 2:
Run the corresponding part with the parameters:
T = 14 (length of time series)
rho_list = the list of the std deviations (e.g [0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2])
distro = "Gauss"


Figure 3 and Figure 6:
Run the corresponding part with the parameters:
T =  length of time series (e.g 12)
N = maximal reservoir dimension (e.g. 100)
MC_steps = Number of Monte Carlo simulations to generate the matrix of moments (e.g. 100000)
sym = "sym" or "iid" (the symmetry condition on the random connectivity matrix)

List of standard deviations:
alpha_list = the list of exponents to be considered in the scalings (e.g. [0.25, 0.5, 0.85, 1.00], corresponding to
                standard deviations [1/N^{0.25}, 1/N^{0.5}, ...]


Figure 4:
Run the corresponding part with the parameters:
T =  length of time series (e.g 12)
rho_list = the list of the std deviations (e.g [0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2])


Figure 5 and Figure 8:
Run the corresponding part with the parameters:
T =  maximal length of time series (e.g 12)
N = reservoir dimension (e.g. 100)
MC_steps = Number of Monte Carlo simulations to generate the matrix of moments (e.g. 100000)
sym = "sym" or "iid" (the symmetry condition on the random connectivity matrix)


Figure 7:
Run the corresponding part with the parameters:
T = length of time series (e.g 12)
rho_list = the list of the std deviations (e.g [0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2])


Figure 9:
Run the corresponding part with the parameters:
T_max = 10 (the max t we consider to plot to)
nb_intervals = 100 (nb_intervals: number of intervals we divide the interval [0,T_max] into)
T_max_2 = 500 (the max t we consider to plot to in subfigure 3)
nb_intervals_2 = 2000 (nb_intervals: number of intervals we divide the interval [0,T_max_2] into)
Choose 3 time series of length 3 each. The ones in the paper are
a = np.array([1.00, 0.0, 0.0])
b = np.array([1.0/math.sqrt(2.0), 0.0, 1.0/math.sqrt(2.0)])
c = np.array([math.sqrt(2.0/3.0), 1.0/math.sqrt(6.0), 1.0/math.sqrt(6.0)])


Figures 10 and 11:
Run the corresponding part with the parameters:
T_max = 10 (the max t we consider to plot to)
nb_intervals = 100 (nb_intervals: number of intervals we divide the interval [0,T_max] into)
a = choice of the time series, e.g. in the paper:
    T = 6 (the length of the time series)
    a= (-1)**np.ogrid[:T+1]/np.array([math.factorial(i) for i in range(T+1)])[::-1]
    a/= np.linalg.norm(a)
connectivity_dim = 10 (reservoir dimension)
MC_steps = 100000 (number of Monte Carlo steps)