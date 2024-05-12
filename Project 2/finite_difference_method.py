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
tau = (h ** 2) / 8
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
alpha = 0.1
u[:, 0] = u0_f(x, x1, x2, c1, c2)
v[:, 0] = v0_f(x, x1, x2, c1, c2)


for j in range(0, m):
    if j - 1 < 0:
        u[j][1] = u[j][0] + (tau * 1j / h ** 2) * (u[m-1][0] - 2 * u[j][0] + u[j+1][0])
        + tau * 1j * (np.abs(u[j][0]) ** 2 + alpha * np.abs(u[j][0]) ** 4 + np.abs(v[j][0]) ** 2) * u[j][0]

        v[j][1] = v[j][0] + (tau * 1j / h ** 2) * (v[m - 1][0] - 2 * v[j][0] + v[j + 1][0])
        + tau * 1j * (np.abs(u[j][0]) ** 2 + np.abs(v[j][0]) ** 2 + alpha * np.abs(v[j][0]) ** 4) * v[j][0]
    elif j + 1 > m - 1:
        u[j][1] = u[j][0] + (tau * 1j / h ** 2) * (u[j-1][0] - 2 * u[j][0] + u[0][0])
        + tau * 1j * (np.abs(u[j][0]) ** 2 + alpha * np.abs(u[j][0]) ** 4 + np.abs(v[j][0]) ** 2) * u[j][0]

        v[j][1] = v[j][0] + (tau * 1j / h ** 2) * (v[j-1][0] - 2 * v[j][0] + v[0][0])
        + tau * 1j * (np.abs(u[j][0]) ** 2 + np.abs(v[j][0]) ** 2 + alpha * np.abs(v[j][0]) ** 4) * v[j][0]
    else:
        u[j][1] = u[j][0] + (tau * 1j / h ** 2) * (u[j - 1][0] - 2 * u[j][0] + u[j + 1][0])
        + tau * 1j * (np.abs(u[j][0]) ** 2 + alpha * np.abs(u[j][0]) ** 4 + np.abs(v[j][0]) ** 2) * u[j][0]

        v[j][1] = v[j][0] + (tau * 1j / h ** 2) * (v[j - 1][0] - 2 * v[j][0] + v[j + 1][0])
        + tau * 1j * (np.abs(u[j][0]) ** 2 + np.abs(v[j][0]) ** 2 + alpha * np.abs(v[j][0]) ** 4) * v[j][0]


for k in tqdm(range(1, n - 1), desc='Running progress', ncols=100):
    for j in range(0, m):
        if j - 1 < 0:
            u[j][k + 1] = (u[j][k-1] + (2 * tau * 1j / h ** 2) * (u[m-1][k] - 2 * u[j][k] + u[j+1][k])
                           + 2 * tau * 1j * (np.abs(u[j][k]) ** 2 + alpha * np.abs(u[j][k]) ** 4 + np.abs(v[j][k]) ** 2) * u[j][k])

            v[j][k + 1] = (v[j][k - 1] + (2 * tau * 1j / h ** 2) * (v[m - 1][k] - 2 * v[j][k] + v[j + 1][k])
                           + 2 * tau * 1j * (np.abs(u[j][k]) ** 2 + alpha * np.abs(v[j][k]) ** 4 + np.abs(v[j][k]) ** 2) * v[j][k])
        elif j + 1 > m - 1:
            u[j][k + 1] = (u[j][k - 1] + (2 * tau * 1j / h ** 2) * (u[j - 1][k] - 2 * u[j][k] + u[0][k])
                           + 2 * tau * 1j * (np.abs(u[j][k]) ** 2 + alpha * np.abs(u[j][k]) ** 4 + np.abs(v[j][k]) ** 2) * u[j][k])

            v[j][k + 1] = (v[j][k - 1] + (2 * tau * 1j / h ** 2) * (v[j - 1][k] - 2 * v[j][k] + v[0][k])
                           + 2 * tau * 1j * (np.abs(u[j][k]) ** 2 + alpha * np.abs(v[j][k]) ** 4 + np.abs(v[j][k]) ** 2) * v[j][k])
        else:
            u[j][k + 1] = (u[j][k - 1] + (2 * tau * 1j / h ** 2) * (u[j - 1][k] - 2 * u[j][k] + u[j + 1][k])
                           + 2 * tau * 1j * (np.abs(u[j][k]) ** 2 + alpha * np.abs(u[j][k]) ** 4 + np.abs(v[j][k]) ** 2) * u[j][k])

            v[j][k + 1] = (v[j][k - 1] + (2 * tau * 1j / h ** 2) * (v[j - 1][k] - 2 * v[j][k] + v[j + 1][k])
                           + 2 * tau * 1j * (np.abs(u[j][k]) ** 2 + alpha * np.abs(v[j][k]) ** 4 + np.abs(v[j][k]) ** 2) * v[j][k])

u_plot = np.abs(u)**2

# Plots a 3D surface
xv, tv = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, tv, u_plot.T)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'$|u|^2$')
plt.show()

def integrate_f(u_values, v_values):
    I = np.zeros(len(t))

    for i in range(0, len(t)):
        I[i] = np.trapz(np.abs(u_values[:, i]) ** 2 + np.abs(v_values[:, i]) ** 2, dx=h)

    return I


conserved = integrate_f(u, v)

error = np.abs(conserved - conserved[0])/conserved[0]

plt.plot(t, error)
plt.show()
