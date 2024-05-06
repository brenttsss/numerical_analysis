import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['text.usetex'] = True

# Brenton Suriah
# 5 May 2024
# Numerical Analysis
# Tutorial 4: Question 2

# Setting up the spatial step and spatial domain
N = 100
L = 10
h = L / N
x = np.arange(-L, L, h)

# Setting up the time domain
t_max = 10
tau = (h**2)/8
t = np.arange(0, t_max + tau, tau)

# Defining an empty array that will eventually store all psi values
m, n = len(x), len(t)
psi = np.zeros((m, n), dtype=complex)


def sech_f(x):
    return 1 / np.cosh(x)


def psi0_f(x, A=1):
    return A * sech_f(A * x) * np.exp(-2 * 1j * x)


# Calculates the initial conditions for the zero-time layer
psi[:, 0] = psi0_f(x)


def mat_first_f(psi_layer):

    """
    Function determines the finite-difference coefficients
    :param psi_layer: column vector of psi
    :return: matrix expressed in terms of psi at k=0
    """

    upper_diagonal = np.diag([(1j*tau)/(h**2)]*(m-1), 1)
    main_diagonal = np.diag([1-2*tau*1j*np.abs(psi_layer[i])**2-(2*1j*tau)/(h**2) for i in range(len(psi_layer))])
    lower_diagonal = np.diag([(1j*tau)/(h**2)]*(m-1), -1)

    val_mat = upper_diagonal + lower_diagonal + main_diagonal

    val_mat[(m-1, 0)] = (1j*tau)/(h**2)
    val_mat[(0, m-1)] = (1j*tau)/(h**2)

    return val_mat


# Calculates the values of psi at the first time layer
psi[:, 1] = np.matmul(mat_first_f(psi[:, 0]), psi[:, 0])


def mat_second_f(psi_layer):

    """
    Function determines the finite-difference coefficients
    :param psi_layer: column vector of psi
    :return: matrix expressed in terms of psi at k=1,...,n
    """

    upper_diagonal = np.diag([(2 * 1j * tau) / (h ** 2)] * (m - 1), 1)
    main_diagonal = np.diag(
        [4 * tau * 1j * np.abs(psi_layer[i])**2 - (4*tau*1j)/(h**2) for i in range(len(psi_layer))])
    lower_diagonal = np.diag([(2 * 1j * tau) / (h ** 2)] * (m - 1), -1)

    val_mat = upper_diagonal + lower_diagonal + main_diagonal

    val_mat[(m - 1, 0)] = (2 * 1j * tau) / (h ** 2)
    val_mat[(0, m - 1)] = (2 * 1j * tau) / (h ** 2)

    return val_mat


# Calculates the values of psi at the second time layer onwards
for k in range(1, n-1):
    psi[:, k + 1] = psi[:, k - 1] + np.matmul(mat_second_f(psi[:, k]), psi[:, k])


# Calculates the probability density
psi = np.abs(psi)**2

# Plots a 3D surface
xv, tv = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, tv, psi.T)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'$|\psi|^2$')
plt.show()
