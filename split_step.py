import numpy as np
from matplotlib import pyplot as plt

# Setting up the spatial step and spatial domain
N = 100
L = 10
h = L / N
x = np.append(np.arange(0, L+h, h), np.arange(-L+h, 0, h))

# Setting up the time domain
t_max = 5
tau = 0.01
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

# Defining array that sort out the exponential bit for the FFT
n = np.append(np.arange(0, N + 1, 1), np.arange(-N + 1, -1 + 1, 1))
exp_val = np.array([np.exp(-1j * (np.pi * n[i] / L) ** 2) for i in range(len(n))])

# Split-step method
for j in range(1, len(t)):
    # Takes the first row of the psi array and copies it
    psi_row = psi[:, j - 1].copy()

    # Calculates the first exponential bit
    psi_val = psi_row * np.exp(2 * 1j * np.abs(psi_row) ** 2 * t[j])

    # Applies the FFT
    psi_val_fft = np.fft.fft(psi_val)

    # Applies the IFFT
    psi[:, j] = np.fft.ifft(exp_val * psi_val_fft)


# Calculates the probability density
psi = np.abs(psi)**2

# Plots a 3D surface
xv, tv = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, tv, psi.T)
plt.show()
