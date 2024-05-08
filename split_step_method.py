import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
plt.rcParams['text.usetex'] = True

N = 200
L = 10
h = L / N
x = np.arange(-L, L, h)

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

n = np.append(np.arange(0, N + 1, 1), np.arange(-N + 1, -1 + 1, 1))

for k in tqdm(range(1, len(t)), desc='Running progress', ncols=100):
    u_col = u[:, k - 1].copy()
    v_col = v[:, k - 1].copy()

    u_val = u_col * np.exp(1j * (np.abs(u_col) ** 2 + alpha * np.abs(u_col) ** 4 + np.abs(v_col) ** 2) * tau)
    v_val = v_col * np.exp(1j * (np.abs(u_col) ** 2 + np.abs(v_col) ** 2 + alpha * np.abs(v_col) ** 4) * tau)

    u_val_fft = np.fft.fft(u_val)
    v_val_fft = np.fft.fft(v_val)

    exp_val = np.exp(-1j * ((np.pi * n / L) ** 2) * tau)

    u_val = np.fft.ifft(exp_val * u_val_fft)
    v_val = np.fft.ifft(exp_val * v_val_fft)

    u[:, k] = u_val
    v[:, k] = v_val

u = np.abs(u) ** 2

# Plots a 3D surface
xv, tv = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, tv, u.T)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'$|u|^2$')
ax.view_init(elev=25, azim=-25)
plt.show()



