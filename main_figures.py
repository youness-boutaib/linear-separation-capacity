from moment_ev_computation import moment_line, moment_matrix, extreme_eigenvalue_list, ev_theoretical_bounds, \
    ev_dominance_N_infty, ev_dominance_T_infty, plot_empirical_proba
from plot_figure import plotDataLog, plotEVBound, plot_ev_N_infty, plot_ev_T_infty, plot_theo_bounds, plot_empirical_proba_HD
import numpy as np
import math


#########################################
#              Figure 1                 #
#########################################
T = 12
N = 100
MC_steps = 100000
sym = "sym"
## Standard deviations: list of exponents
alpha_list =[0.25, 0.5, 0.75, 1.00]

B = ev_dominance_N_infty(T, N, sym, alpha_list, MC_steps)
plot_ev_N_infty(N,alpha_list,B)

#########################################
#              Figure 2                 #
#########################################
T = 12
rho_list = [0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2]

distro = "Wigner_sc"
Xalpha_list=[]
B = moment_line(T, distro, 1.0)
A = moment_matrix(B , T)
i, j = np.meshgrid(np.arange(T + 1), np.arange(T + 1), indexing='ij')
for rho in rho_list:
    C = (rho**(i+j))*A
    X = ev_theoretical_bounds(T, C)
    Xalpha_list.append([rho,X])

plotEVBound(Xalpha_list, T, 1)

#########################################
#              Figure 3                 #
#########################################
T = 12
N = 100
MC_steps = 100000
sym = "sym"
## Standard deviations: list of exponents
alpha_list =[0.25, 0.5, 0.75, 1.00]

C = ev_dominance_T_infty(T, N, sym, alpha_list, MC_steps)
plot_ev_T_infty(T,alpha_list, C)

#########################################
#              Figure 4                 #
#########################################
T = 12
N = 100
MC_steps = 100000
sym = "iid"
## Standard deviations: list of exponents
alpha_list =[0.25, 0.5, 0.75, 1.00]

B = ev_dominance_N_infty(T, N, sym, alpha_list, MC_steps)
plot_ev_N_infty(N,alpha_list,B)

#########################################
#              Figure 5                 #
#########################################
T = 12
rho_list = [0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2]

Xalpha_list=[]
for rho in rho_list:
    A = np.diag(rho**(2*np.arange(T+1)))
    X = ev_theoretical_bounds(T, A)
    Xalpha_list.append([rho,X])

plotEVBound(Xalpha_list, T, 1)

#########################################
#              Figure 6                 #
#########################################
T = 12
N = 100
MC_steps = 75000
sym = "iid"
## Standard deviations: list of exponents
alpha_list =[0.15, 0.25, 0.5, 0.75, 0.85, 1.00]

C = ev_dominance_T_infty(T, N, sym, alpha_list, MC_steps)
plot_ev_T_infty(T,alpha_list, C)