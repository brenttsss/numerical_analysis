import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True


def f(y, x):
    return y[1], 2 * y[0] ** 3


def shooting_method_f(p, x, tol):
    pstart, pend = p
    diff = np.abs(pend - pstart)
    n_iterations = 0

    while diff > tol:
        updated_p = (pend + pstart) / 2
        y0 = [updated_p, -1 / 4]
        y = odeint(f, y0, x)

        if y[-1][0] + y[-1][1] - 4 / 25 < 0:
            pstart = updated_p
        elif y[-1][0] + y[-1][1] - 4 / 25 > 0:
            pend = updated_p

        diff = np.abs(pend - pstart)
        n_iterations += 1

    return [y[:, 0], n_iterations]


def plot_f(x, y_ansatz, y, n_iterations):
    fig, ax = plt.subplots()
    ax.plot(x, y_ansatz, color='red', label=r'$y_\mathrm{ansatz}$')
    ax.plot(x, y, color='black', label=r'$y$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    if n_iterations == 1:
        ax.set_title(r'Newton-Kantorovich Method, {} iteration'.format(n_iterations))
    else:
        ax.set_title(r'Newton-Kantorovich Method, {} iterations'.format(n_iterations))
    ax.legend()
    return plt.show()






