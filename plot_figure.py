from matplotlib import pyplot as plt
import numpy as np
import string
from moment_ev_computation import fake_log_theo_curve_bound, empirical_proba_concentration_HD


# X: 2 x L+1  tensor of eigenvalues
# L integer
def plotDataLog(X, L):
    X_log = np.log(X)
    L_numpy = np.arange(L+1)
    plt.plot(L_numpy, X_log[0,:], label="$log(\lambda_{min})$")
    plt.plot(L_numpy, X_log[1,:], label="$log(\lambda_{max})$")
    # this is to add the labels
    plt.legend()
    plt.xlabel("length T of the sequence")
    plt.ylabel("$\log(\lambda)$")
    plt.show()


# Xalpha_list is a list of [alpha, X]
# X: is a 3x(T+1) tensor:
# #       first line = lower bound from 0 to T
# #       second line = lambda_max from 0 to T
# #       third line = upper bound from 0 to T
# T integer (for time)
# This function plots
# n is either 0 or 1:
# n = 0 we plot log( fake dominance / true dominance)
# n = 1 we plot true dominance
def plotEVBound(Xalpha_list, T, n):
    L_numpy = np.arange(T + 1)
    # Case 1: we want to plot log( fake dominance / true dominance)
    if n == 0:
        for Z in Xalpha_list:
            # Recover the X
            X=Z[1]
            log_ratio = np.log(X[0]/X[1])
            plt.plot(L_numpy, log_ratio, label=r"$\rho = $"+str(Z[0]))
            plt.ylabel(r'$\log\left( \frac{\widetilde{r}_T}{r_T} \right)$')
    # Case 2: we want to plot true dominance
    else:
        for Z in Xalpha_list:
            # Recover the X
            X = Z[1]
            dominance = X[1]/X[2]
            plt.plot(L_numpy, dominance, label=r"$\rho = $"+str(Z[0]))
            plt.ylabel("dominance of the largest eigenvalue")
    # this is to add the labels
    plt.legend()
    plt.xlabel("length T of the sequence")
    plt.show()


# N_max is the largest considered dimension of the reservoir
# alpha is the list of the considered exponents 1/N^{alpha} of length q
# dominance array is a qxN_max array of dominance ratios:
#       each row for a fixed considered scaling, going from n=1 to N_max
# Plots evolution of dominance for different scalings
def plot_ev_N_infty(N_max, alpha_list, dominance_array):
    L_numpy = np.arange(1, N_max + 1)
    for alpha_index in range(len(alpha_list)):
        p_str = "{:.2f}".format(alpha_list[alpha_index])
        plt.plot(L_numpy, dominance_array[alpha_index,:], label=r"$\rho = 1/N^{"+p_str+"}$")

    # this is to add the labels
    plt.legend()
    plt.xlabel("dimension N of the reservoir")
    plt.ylabel("$\lambda_{max}/$"+"spectrum")
    plt.show()


# T is the largest time considered
# alpha is the list of the considered exponents 1/N^{alpha} of length q
# dominance array is a qx(T+1) array of dominance ratios:
#       each row for a fixed considered scaling, going from t=0 to T
# Plots evolution of dominance for different scalings
def plot_ev_T_infty(T, alpha_list, dominance_array):
    L_numpy = np.arange(T + 1)
    for alpha_index in range(len(alpha_list)):
        p_str = "{:.2f}".format(alpha_list[alpha_index])
        if alpha_list[alpha_index]!=0:
            plt.plot(L_numpy, dominance_array[alpha_index,:], label=r"$\rho = 1/N^{"+p_str+"}$")
        else:
            plt.plot(L_numpy, dominance_array[alpha_index, :], label=r"$\rho = 1$")

    # this is to add the labels
    plt.legend()
    plt.xlabel("length T of the sequence")
    plt.ylabel("$\lambda_{max}/$"+"spectrum")
    plt.show()


# T number for max length of time
# list_of_a_0 gives the list of leading terms in the time series
# the function draws the theoretical tail bounds for different values of a_0
def plot_theo_bounds(T_max, nb_intervals, list_of_a):
    # t_list is the np_list of numbers t we want to evaluate at = [t_0, ... , t_{nb_intervals}]
    t_list = np.linspace(0, T_max, nb_intervals + 1)
    alphabet = list(string.ascii_lowercase)
    key = 0
    for a in list_of_a:
        y_values = fake_log_theo_curve_bound(a, T_max, nb_intervals)
        letter = alphabet[key]
        key = key +1
        plt.plot(t_list, y_values, label=f'${{\\mathbf{{{letter}}}}}$')

    # this is to add the labels
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("$-\eta(t)$")
    plt.show()


# a = (a_0, ... , a_T) is the time-series (T+1) 1-dim numpy
# connectivity_dim is the dimension of the reservoir
# t_max number for length of time
# the function draws the density function then the tail
def plot_empirical_proba_HD(a, T_max, nb_intervals, connectivity_dim, MC_steps):
    # t_list is the np_list of numbers t we want to evaluate at = [t_0, ... , t_{nb_intervals}]
    t_list = np.linspace(0, T_max, nb_intervals + 1)
    result_0 = empirical_proba_concentration_HD(a, T_max, nb_intervals, "sym", connectivity_dim, MC_steps)
    result_1 = empirical_proba_concentration_HD(a, T_max, nb_intervals, "iid", connectivity_dim, MC_steps)
    # plot density first
    y_values = result_0[0, :]
    plt.plot(t_list, y_values, label="symmetric")
    y_values = result_1[0, :]
    plt.plot(t_list, y_values, label="i.i.d.")
    # this is to add the labels
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Density of $\|\|f^2(\mathbf{a},W)\|\|^2$")
    plt.show()

    # plot deviation from mean
    y_values = np.log(result_0[1, :])
    plt.plot(t_list, y_values, label="symmetric")
    y_values = np.log(result_1[1, :])
    plt.plot(t_list, y_values, label="i.i.d.")
    # this is to add the labels
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("$\log(\mathbb{P}(\|\|f^2(\mathbf{a},W)-\mathbb{E}f^2(\mathbf{a},W)\|\|\geq t))$")
    plt.show()
