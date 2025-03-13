import numpy as np
from matplotlib import pyplot as plt
import string
import math


# T is an integer
# distro is a string that specifies which proba distro we want to use
# alpha is a real number by which the distro is multiplied
# The function returns the distro moment from order 0 to 2T as a 1x(2T+1) tensor
def moment_line(T, distro, alpha =1.00):
    if T == 0:
        return np.ones((1,1))
    else:
        line = moment_line(T - 1, distro, alpha)
        if distro == "Gauss":
            return np.concatenate((line, np.array([[0, line[0 , 2 * (T - 1) ] * (2 * T - 1) * (alpha ** 2)]])), axis=1)
        elif distro == "Rademacher":
            return np.concatenate((line, np.array([[0, alpha ** (2 * T)]])), axis=1)
        elif distro == "Uniform":
            return np.concatenate((line,
                              np.array([[ alpha ** (2 * T - 1) / (2 * T), alpha ** (2 * T) / (2 * T + 1)]]))
                             , axis=1)
        elif distro == "Wigner_sc":
            return np.concatenate((line,
                                   np.array([[0, 2*line[0 , 2 * (T - 1)]*(alpha**2)*(2 * T-1) / (T + 1)]]))
                                  , axis=1)


# T is an integer
# momentline is the tensor of moments of the distribution from 0 to the largest 2T: a 1x(2T+1) tensor
# The function returns the distro moment matrix up to order 2T as a T+1 square tensor
def moment_matrix(momentline, T):
    if T == 0:
        return np.ones((1,1))
    else:
        # first moments to concatenate (last row): from T to 2T-1
        needed_moments_1 = momentline[:, T: (2*T)]
        # construct a (T+1) x T tensor
        first_concat = np.concatenate((moment_matrix(momentline,T - 1), needed_moments_1)
                                 , axis=0)
        # last moments to concatenate as a column: from T to 2T
        needed_moments_2 = momentline[:, T:(2*T+1)].T
        # construct the final 2(T+1) square tensor
        final_matrix = np.concatenate((first_concat , needed_moments_2)
                                 , axis=1)
        return final_matrix


# A is (T+1)x(T+1) square tensor of some moments
# T is an integer
# Function returns 2 x (T+1) tensor: first row for lambda_min, second row for lambda_max
def extreme_eigenvalue_list(A, T):
    if T == 0:
        return np.ones((2,1))
    else:
        # compute eigenvalues of the global matrix
        spectrum = np.linalg.eigvalsh(A)
        new_eigenvalues = np.array([[np.min(spectrum)], [np.max(spectrum)]])
        # extract TxT square tensor of lower moments
        extracted_matrix = A[:T,:T]
        # we concatenate the new results with the previous ones
        return np.concatenate((extreme_eigenvalue_list(extracted_matrix, T - 1), new_eigenvalues), axis=1)


# T is an integer
# momentmatrix distro moment matrix up to order 2T as a T+1 square tensor
# The function returns a 3x(T+1) tensor:
#       first line = lower bound from 0 to T
#       second line = lambda_max from 0 to T
#       third line = upper bound from 0 to T
def ev_theoretical_bounds(T, momentmatrix):
    if T == 0:
        return np.ones((3,1))
    else:
        # the smallest bound is the largest l_2 norm of rows
        row_norms = np.linalg.norm(momentmatrix, axis=1)
        small_bound = np.max(row_norms)
        # the largest bound is the sum of the diagonal elements
        large_bound = np.trace(momentmatrix)
        # compute eigenvalues of the global matrix
        spectrum = np.linalg.eigvalsh(momentmatrix)
        new_bounds = np.array([[small_bound],[np.max(spectrum)], [large_bound]])
        # new_momentmatrix distro moment matrix up to order 2(L-1) as an 2L square tensor
        new_momentmatrix = momentmatrix[0:T,0:T]
        # we concatenate the new results with the previous ones
        return np.concatenate((ev_theoretical_bounds(T - 1,  new_momentmatrix), new_bounds), axis=1)


# returns global matrix of u's : N(T+1) x (T+1) pytoch tensor [[u,0,...,0], [0,u,...,0], ... , [0,0,...,u]]
# Useful to compute beta matrix (then B) using matrix multiplication
def construct_u(T, N):
    u = np.zeros((N*(T+1), T+1))
    for t in range(T+1):
        u[t*N:(t+1)*N, t] = np.ones(N)
    return u


# sym = sym or iid
class Connectivity_List:
    # We later want this to become N x N(T+1) pytorch tensor [I, W, W^2, ..., W^T]
    def __init__(self, T, N, sigma, sym):
        # We start with the first element I
        self.matrix_power_list=np.eye(N,N)
        if T >0:
            W_powered = np.eye(N, N)
            W = np.random.normal(0, sigma, (N, N))
            if sym == "sym":
                W=np.triu(W) + np.triu(W, 1).T
            for t in range(1, T+1):
                # compute W^t : N x N pytorch tensor
                W_powered = np.matmul(W_powered, W)
                # this is now : N x N(t+1) pytorch tensor [I, W, W^2, ..., W^t]
                self.matrix_power_list = np.concatenate((self.matrix_power_list, W_powered), axis =1)

        # global matrix of u's : N(T+1) x (T+1) pytoch tensor
        u = construct_u(T, N)
        # beta matrix: N x (T+1) pytoch tensor (each row for the first index, column for the length of multiplication)
        # beta matrix: [I, Wu, W^2u,...,W^Tu]
        self.beta = np.matmul(self.matrix_power_list,u)
        # B matrix: (T+1) x (T+1) pytoch tensor (to be used for Monte Carlo scheme)
        self.B = np.matmul((self.beta).T, self.beta)


# returns the true Hankel matrix constructed via Monte Carlo method: (T+1) x (T+1) square pytorch matrix
def monte_carlo_matrix_quadratic(T, N, sigma, sym, MC_steps):
    sum_mat = np.zeros((T+1,T+1))
    for i in range(MC_steps):
        sum_mat = sum_mat + Connectivity_List(T, N, sigma, sym).B
    return sum_mat/MC_steps


# B_mat plays the role of B but can be of any dimension. We assume it is built with scaling 1/N^{0.5}
# N is the dimension of the reservoir
# alpha is the new scaling -> 1/N^{alpha}
# returns matrix of same dimension as B_mat with scaling 1/N^{alpha}
def matrix_rescaling(Bmat, N, alpha):
    T_1, T_2 = Bmat.shape
    i, j = np.ogrid[:T_1, :T_2]
    correction_term = N**(0.5-alpha)
    powers = correction_term ** (i + j)
    return powers*Bmat


# T is an integer
# N_max is the largest considered dimension of the reservoir
# sym = sym or iid
# alpha is the list of the considered rescalings 1/N^{alpha} (of length q)
# The function returns a qxN_max array of dominance ratios:
#       each row for a fixed considered scaling, going from n=1 to N_max
def ev_dominance_N_infty(T, N_max, sym, alpha_list, MC_steps):
    dominance_array = np.zeros((len(alpha_list), N_max))
    for n in range(1,N_max+1):
        #construct Generalised Moments Matrix (T+1)x(T+1) for the base scaling 1/N^{0.5}
        gen_moment_matrix_std = monte_carlo_matrix_quadratic(T, n, 1/(n**0.5), sym, MC_steps)
        print("gen_moment_matrix_std done for reservoir dimension", n)
        for alpha_index in range(len(alpha_list)):
            # construct Hankel matrix (T+1)x(T+1) for the scaling 1/N^{alpha}
            gen_moment_matrix_rescaled = matrix_rescaling(gen_moment_matrix_std, n, alpha_list[alpha_index])
            # compute eigenvalues
            spectrum = np.linalg.eigvalsh(gen_moment_matrix_rescaled)
            # retain the dominance ratio, n-1 because we start from 1
            dominance_array[alpha_index,n-1]=np.max(spectrum)/np.trace(gen_moment_matrix_rescaled)
    return dominance_array


# T is the largest time considered
# N is the dimension of the reservoir
# rho is the std dev, usually = 1/N^{0.5}
# sym = sym or iid
# alpha is the list of the considered scalings 1/N^{alpha} (of length q)
# small_bound="yes" or "no" to compute the (2,\inf)-norm/trace lower bound
# The function returns a qx(T+1) array of dominance ratios:
#       each row for a fixed considered scaling, going from t=0 to T
# If small_bound="yes", we return a second qxN_max array of lower dominance ratios
def ev_dominance_T_infty(T, N, sym, alpha_list, MC_steps, small_bound="no"):
    dominance_array = np.zeros((len(alpha_list), T+1))
    if small_bound =="yes":
        small_dominance_array = np.zeros((len(alpha_list), T+1))

    # construct gen_moment_matrix_std matrix (T+1)x(T+1) for the base scaling 1/N^{0.5}
    gen_moment_matrix_std = monte_carlo_matrix_quadratic(T, N, 1/(N**0.5), sym, MC_steps)
    # construct the list of gen_moment_matrix_std (T+1)x(T+1)  matrices for all scalings (length same as alpha)
    gen_moment_matrix_rescaled = []
    for alpha_index in range(len(alpha_list)):
        gen_moment_matrix_rescaled.append(matrix_rescaling(gen_moment_matrix_std, N, alpha_list[alpha_index]))

    for t in range(T + 1):
        for alpha_index in range(len(alpha_list)):
            # Deduce gen_moment_matrix_std matrix (t+1)x(t+1) for the scaling alpha
            gen_moment_matrix_truncated = (gen_moment_matrix_rescaled[alpha_index])[:t+1,:t+1]
            # compute eigenvalues
            spectrum = np.linalg.eigvalsh(gen_moment_matrix_truncated)
            # retain the dominance ratio
            dominance_array[alpha_index,t]=np.max(spectrum)/np.trace(gen_moment_matrix_truncated)
            if small_bound == "yes":
                small_dominance_array[alpha_index, t] = np.max(np.linalg.norm(gen_moment_matrix_truncated, axis=1))/np.trace(gen_moment_matrix_truncated)
            # return dominance_array
    if small_bound == "yes":
        return [dominance_array, small_dominance_array]
    else:
        return dominance_array


# T_max is the max t we consider to plot to
# nb_intervals: number of intervals we divide the interval [0,T] into
# a = (a_0, ... , a_T) is the time-series (T+1) 1-dim numpy with T=2
# Function returns a 2 x (N+1) matrix:
#   first row: the histogram (density) of f_squared
#   second row: the tail probas f^2-Ef^2
def empirical_proba_concentration_forloop(a, T_max, nb_intervals):
    # length of sub-intervals
    delta = T_max / nb_intervals
    # t_list is the np_list of numbers t we want to evaluate at = [t_0, ... , t_{nb_intervals}]
    t_list = np.linspace(0, T_max, nb_intervals + 1)
    # add a final step t_extended : 1 x (length of time +1)
    t_extended = np.concatenate((t_list, [t_list[nb_intervals] + delta])).reshape(1, -1)

    # number of Monte Carlo samples
    nb_batches = 100000
    batch_size = 100
    nb_trials = nb_batches * batch_size

    # Expectation of the (square of) reservoir state
    expectation = 3 * a[0] ** 2 + a[1] ** 2 + 2 * a[0] * a[2] + a[2] ** 2
    # What we aim to return
    histogram_mean = np.zeros((1, nb_intervals + 1))
    compare_matrix_mean = np.zeros((1, nb_intervals + 1))
    for i in range(nb_batches):
        # Create a list of simulated Gaussians: 1 x batch_size
        w_simulated = np.random.randn(batch_size).reshape((1, batch_size))
        # stack the powers: first row w^2, second w then 1's. 3 x batch_size np array
        w_powers = np.concatenate((w_simulated ** 2,
                                   np.concatenate((w_simulated, np.ones((1, batch_size))), axis=0)
                                   ), axis=0)
        # compute f^2: 1 x batch_size numpy. Each entry for the randomly picked w
        f_squared = (np.matmul(a.reshape(1, 3), w_powers)) ** 2
        # Histogram "f_squared in [t_i, t_{i+1}]"? 1 x nb_trials numpy of 0 or 1s
        # Create the histogram by comparison: (nb_intervals+1) x batch_size numpy of 0 or 1s
        histogram = ((t_extended[:, 0:-1].T <= f_squared) & (t_extended[:, 1:].T > f_squared)).astype(float)
        # Create the histogram: 1x (nb_intervals+1) numpy of sums of 0s and 1s
        # (Entries = nb times f_squared was in the subinterval)
        histogram_mean += np.sum(histogram, axis=1).reshape(1, -1)

        # deviation_from_mean= |distance - expectation| to the time: 1 x batch_size numpy
        deviation_from_mean = np.abs(f_squared - expectation)
        # compare the |distance - expectation| to the time: (nb_intervals+1) x batch_size numpy of 0 or 1s
        compare_matrix = (deviation_from_mean > t_list.reshape((len(t_list), 1))).astype(float)
        # Create the tail: 1x (T+1) numpy of 0 or 1s
        compare_matrix_mean += np.sum(compare_matrix, axis=1).reshape(1, -1)
    return np.concatenate((histogram_mean / (nb_trials * delta), compare_matrix_mean / nb_trials), axis=0)


# T_max is the max t we consider to plot to
# nb_intervals: number of intervals we divide the interval [0,T] into
# list_of_a gives the time series, each as (T+1) 1-dim numpy with T=2
# the function draws the density function then the tail
def plot_empirical_proba(T_max, nb_intervals, list_of_a):
    # t_list is the np_list of numbers t we want to evaluate at = [t_0, ... , t_{nb_intervals}]
    t_list = np.linspace(0, T_max, nb_intervals + 1)
    alphabet = list(string.ascii_lowercase)
    results = []
    for a in list_of_a:
        results.append(empirical_proba_concentration_forloop(a, T_max, nb_intervals))

    key = 0
    for k in range(len(list_of_a)):
        z_values = results[k][0, :]
        letter = alphabet[key]
        key = key + 1
        plt.plot(t_list, z_values, label=f'${{\\mathbf{{{letter}}}}}$')

    # this is to add the labels
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Density of $f^2(\mathbf{x},w)$")
    plt.show()

    key = 0
    for k in range(len(list_of_a)):
        y_values = np.log(results[k][1, :])
        letter = alphabet[key]
        key = key + 1
        plt.plot(t_list, y_values, label=f'${{\\mathbf{{{letter}}}}}$')

    # this is to add the labels
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("$\log(\mathbb{P}(|f^2(\mathbf{x},w)-\mathbb{E}f^2(\mathbf{x},w)|\geq t))$")
    plt.show()


# a=[a_0,a_1,a_2] is a time series, np list of length 3
# T_max is the max t we consider to plot to
# nb_intervals: number of intervals we divide the interval [0,T] into
# Function return the (log = -eta) theoretical concentration bound for all t's based on a_0 being the leading term
def fake_log_theo_curve_bound(a, T_max, nb_intervals):
    # t_list is the np_list of numbers t we want to evaluate at = [t_0, ... , t_{nb_intervals}]
    t_list = np.linspace(0, T_max, nb_intervals + 1)

    moments = []
    moments.append(3 * a[0] ** 2 + a[1] ** 2 + 2 * a[0] * a[2] + a[2] ** 2)
    moments.append(6 * a[0] * a[1] + 2 * a[1] * a[2])
    moments.append(12 * a[0] ** 2 + 2 * a[1] ** 2 + 4 * a[0] * a[2])
    moments.append(12 * a[0] * a[1])
    moments.append(24 * a[0] ** 2)

    for p in range(1, 5):
        for k in range(p, 5):
            if p == 1 and k == 1:
                eta_t = (t_list / moments[1]) ** (2)
            else:
                eta_t = np.fmin(eta_t, (t_list / moments[k]) ** (2 / p))

    return -eta_t


# T_max is the max t we consider to plot to
# nb_intervals: number of intervals we divide the interval [0,T] into
# a = (a_0, ... , a_T) is the time-series (T+1) 1-dim numpy list
# sym = sym or iid
# connectivity_dim is the dimension of the reservoir
# Function returns a 2 x (p+1) matrix:
#   first row: the histogram/density of f_squared
#   second row: the tail probas f^2-Ef^2
def empirical_proba_concentration_HD(a, T_max, nb_intervals, sym, connectivity_dim, MC_steps):
    # length of sub-intervals
    delta = T_max / nb_intervals
    # t_list is the np_list of numbers t we want to evaluate at = [t_0, ... , t_{nb_intervals}]
    t_list = np.linspace(0, T_max, nb_intervals + 1)
    # T+1 = length of time series
    T = len (a) - 1
    rho = 1 / math.sqrt(connectivity_dim)

    # What we aim to return
    histogram_mean = np.zeros((1,nb_intervals+1))
    tail_mean = np.zeros((1,nb_intervals+1))
    # We will need reversed_a to compute the distance
    reversed_a = a[::-1].reshape(1, -1)
    # We will need the expected values to compute the proba of deviation from them
    # returns the true Hankel matrix constructed via Monte Carlo method: (T+1) x (T+1) square pytorch matrix
    expected_B_T_N = monte_carlo_matrix_quadratic(T, connectivity_dim, rho, sym, MC_steps)
    expected_f_squared = np.matmul(np.matmul(reversed_a.reshape(1,T+1), expected_B_T_N), reversed_a.reshape(T+1,1))
    for i in range(MC_steps):
        # Simulate B_T_N: (T+1)x(T+1) np array
        B_T_N = Connectivity_List(T, connectivity_dim, rho, sym).B
        # compute simulated |f^2|: float number
        f_squared = np.matmul(np.matmul(reversed_a.reshape(1,T+1), B_T_N), reversed_a.reshape(T+1,1))
        # Histogram "f_squared in [t_i, t_{i+1}]"? 1 x nb_trials numpy of 0 or 1s
        # add a final step t_extended : 1 x (nb_intervals +2)
        t_extended = np.concatenate((t_list, [t_list[nb_intervals] + delta])).reshape(1, -1)
        # Create the histogram by comparison: 1 x (p+1) numpy of 0 or 1s
        histogram_mean += ((t_extended[:,0:-1]<= f_squared) & (t_extended[:,1:]>f_squared)).astype(float)
        # deviation_from_mean= |distance - expectation| to the time: float number
        deviation_from_mean = np.abs(f_squared - expected_f_squared)
        # compare the |distance - expectation| to the time: 1 x (p+1)
        tail_mean += (deviation_from_mean > t_list.reshape((1,len(t_list)))).astype(float)
    return np.concatenate((histogram_mean/(MC_steps*delta), tail_mean/MC_steps),axis=0)

