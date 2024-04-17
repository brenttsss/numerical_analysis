import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True

# Brenton Suriah
# 17 April 2024
# Numerical Analysis
# Tutorial 4: Question 1

# Setting up the spatial step and spatial domain
N = 1000
L = 10
h = L / N
x = np.arange(-L, L, h)

# Setting up the time domain
t_max = 10
tau = 0.025
t = np.arange(0, t_max + tau, tau)

# Defining an empty array that will eventually store all psi values
psi = np.zeros((len(x), len(t)), dtype=complex)


def sech(x):
    """Calculates the value of sech(x)"""
    return 1 / np.cosh(x)


def psi_0(x, A=1):
    """Calculates the initial values of the soliton"""
    return A * sech(A * x) * np.exp(-2 * 1j * x)


# Calculates and stores the initial values into the psi array
psi[:, 0] = psi_0(x)

# Defining arrays that sorts out the exponential bit for the FFT
n = np.append(np.arange(0, N + 1, 1), np.arange(-N + 1, -1 + 1, 1))

# Split-step method
for j in range(1, len(t)):
    psi_row = psi[:, j - 1].copy()

    psi_val = psi_row * np.exp(2j * (np.abs(psi_row) ** 2) * tau)

    psi_val_fft = np.fft.fft(psi_val)

    exp_val = np.exp(-1j * ((np.pi * n / L) ** 2) * tau)

    psi_val = np.fft.ifft(exp_val * psi_val_fft)

    psi[:, j] = psi_val


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
