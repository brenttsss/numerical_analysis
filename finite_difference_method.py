import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
plt.rcParams['text.usetex'] = True

N = 200
L = 10
h = L / N
x = np.arange(-L, L+h, h)

t_max = 5
tau = (h ** 2) / 6
t = np.arange(0, t_max + tau, tau)

m, n = len(x), len(t)
u = np.zeros((m, n), dtype=complex)
v = np.zeros((m, n), dtype=complex)


def sech_f(x):
    return 1 / np.cosh(x)


def u0_f(x, x1, x2, c1, c2):
    term_1 = sech_f(x - x1) * np.exp(1j * c1 * (x - x1))
    term_2 = sech_f(x - x2) * np.exp(1j * c2 * (x - x2))
    return term_1 + term_2


def v0_f(x, x1, x2, c1, c2):
    term_1 = sech_f(x - x1) * np.exp(1j * c1 * (x - x1))
    term_2 = sech_f(x - x2) * np.exp(1j * c2 * (x - x2))
    return term_1 + term_2


x1, x2, c1, c2 = -5, 5, 1, -1
alpha = -0.001
u[:, 0] = u0_f(x, x1, x2, c1, c2)
v[:, 0] = v0_f(x, x1, x2, c1, c2)


def layer_matrix_u_f(u_layer, v_layer):
    upper_diagonal = np.diag([2 * tau * 1j / (h ** 2)] * (m - 1), 1)
    lower_diagonal = np.diag([2 * tau * 1j / (h ** 2)] * (m - 1), -1)
    main_diagonal = np.diag([-4 * tau * 1j / (h ** 2) + 2 * tau * 1j * (
                np.abs(u_layer[i]) ** 2 + alpha * np.abs(u_layer[i]) ** 4 + np.abs(v_layer[i]) ** 2) for i in
                             range(len(u_layer))])

    val_mat = upper_diagonal + lower_diagonal + main_diagonal
    val_mat[(m - 1, 0)] = (2 * 1j * tau) / (h ** 2)
    val_mat[(0, m - 1)] = (2 * 1j * tau) / (h ** 2)

    return val_mat


def layer_matrix_v_f(u_layer, v_layer):
    upper_diagonal = np.diag([2 * tau * 1j / (h ** 2)] * (m - 1), 1)
    lower_diagonal = np.diag([2 * tau * 1j / (h ** 2)] * (m - 1), -1)
    main_diagonal = np.diag([-4 * tau * 1j / (h ** 2) + 2 * tau * 1j * (
                np.abs(u_layer[i]) ** 2 + np.abs(v_layer[i]) ** 2 + alpha * np.abs(v_layer[i]) ** 4) for i in
                             range(len(v_layer))])

    val_mat = upper_diagonal + lower_diagonal + main_diagonal
    val_mat[(m - 1, 0)] = (2 * 1j * tau) / (h ** 2)
    val_mat[(0, m - 1)] = (2 * 1j * tau) / (h ** 2)

    return val_mat


def first_layer_matrix_u_f(u_value, v_value):
    upper_diagonal = np.diag([tau * 1j / (h ** 2)] * (m - 1), 1)
    lower_diagonal = np.diag([tau * 1j / (h ** 2)] * (m - 1), -1)
    main_diagonal = np.diag([1 - (2 * tau * 1j / (h ** 2) + tau * 1j * (
                np.abs(u_value[i]) ** 2 + alpha * np.abs(u_value[i]) ** 4 + np.abs(v_value[i]) ** 2)) for i in
                             range(len(u_value))])

    val_mat = upper_diagonal + lower_diagonal + main_diagonal
    val_mat[(m - 1, 0)] = (1j * tau) / (h ** 2)
    val_mat[(0, m - 1)] = (1j * tau) / (h ** 2)

    return val_mat


def first_layer_matrix_v_f(u_value, v_value):
    upper_diagonal = np.diag([tau * 1j / (h ** 2)] * (m - 1), 1)
    lower_diagonal = np.diag([tau * 1j / (h ** 2)] * (m - 1), -1)
    main_diagonal = np.diag([1 - (2 * tau * 1j / (h ** 2) + tau * 1j * (
                np.abs(u_value[i]) ** 2 + np.abs(v_value[i]) ** 2 + alpha * np.abs(v_value[i]) ** 4)) for i in
                             range(len(v_value))])

    val_mat = upper_diagonal + lower_diagonal + main_diagonal
    val_mat[(m - 1, 0)] = (1j * tau) / (h ** 2)
    val_mat[(0, m - 1)] = (1j * tau) / (h ** 2)

    return val_mat


u[:, 1] = np.matmul(first_layer_matrix_u_f(u[:, 0], v[:, 0]), u[:, 0])
v[:, 1] = np.matmul(first_layer_matrix_u_f(u[:, 0], v[:, 0]), v[:, 0])

for k in tqdm(range(1, n - 1), desc='Running progress', ncols=100):
    u[:, k + 1] = u[:, k - 1] + np.matmul(layer_matrix_u_f(u[:, k], v[:, k]), u[:, k])
    v[:, k + 1] = v[:, k - 1] + np.matmul(layer_matrix_v_f(u[:, k], v[:, k]), v[:, k])

u = np.abs(u)**2

# Plots a 3D surface
xv, tv = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, tv, u.T)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'$|u|^2$')
plt.show()
