from moment_ev_computation import moment_line, moment_matrix, extreme_eigenvalue_list, ev_theoretical_bounds, \
    ev_dominance_N_infty, ev_dominance_T_infty, plot_empirical_proba
from plot_figure import plotDataLog, plotEVBound, plot_ev_N_infty, plot_ev_T_infty, plot_theo_bounds, plot_empirical_proba_HD
import numpy as np
import math


#########################################
#              Figure 1                 #
#########################################
T = 10
alpha = 1.00
distro = "Gauss"

# The distro moment from order 0 to 2T as an 1x(2T+1) tensor
B = moment_line(T, distro, alpha)
# The distro moment matrix up to order 2T as a T+1 square tensor
A = moment_matrix(B , T)
# List of eigenvalues: first row for lambda_min, second row for lambda_max
C= extreme_eigenvalue_list(A, T)
# Plot the evolution of the logarithms of the eigenvalues
plotDataLog(C, T)


#########################################
#              Figure 2                 #
#########################################
T = 14
rho_list = [0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2]
distro = "Gauss"

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

B = ev_dominance_N_infty(T, N, sym, alpha_list, MC_steps)
plot_ev_N_infty(N,alpha_list,B)

#########################################
#              Figure 4                 #
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
#              Figure 5                 #
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
#              Figure 6                 #
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
#              Figure 7                 #
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
#              Figure 8                 #
#########################################
T = 12
N = 100
MC_steps = 75000
sym = "iid"
## Standard deviations: list of exponents
alpha_list =[0.15, 0.25, 0.5, 0.75, 0.85, 1.00]

C = ev_dominance_T_infty(T, N, sym, alpha_list, MC_steps)
plot_ev_T_infty(T,alpha_list, C)


#########################################
#              Figure 9                 #
#########################################
T_max = 10
nb_intervals = 100
T_max_2 = 500
nb_intervals_2 = 2000
a = np.array([1.00, 0.0, 0.0])
b = np.array([1.0/math.sqrt(2.0), 0.0, 1.0/math.sqrt(2.0)])
c = np.array([math.sqrt(2.0/3.0), 1.0/math.sqrt(6.0), 1.0/math.sqrt(6.0)])

list_of_a = [a,b,c]
plot_empirical_proba(T_max, nb_intervals, list_of_a)
plot_theo_bounds(T_max_2, nb_intervals_2, list_of_a)


#########################################
#          Figures 10 and 11            #
#########################################
T_max = 10
nb_intervals = 100
connectivity_dim = 50
MC_steps = 100000

T = 5
a= (-1)**np.ogrid[:T+1]/np.array([math.factorial(i) for i in range(T+1)])[::-1]
a/= np.linalg.norm(a)

plot_empirical_proba_HD(a, T_max, nb_intervals, connectivity_dim, MC_steps)