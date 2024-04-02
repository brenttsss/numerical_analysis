import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import solve

h, k = 0.5, 0.0025
alpha = 1
xstart, xstop = [-10, 10]
tstart, tstop = [0, 40]

x = np.arange(xstart, xstop + h, h)
t = np.arange(tstart, tstop + k, k)
m, n = len(x), len(t)

boundary_conditions = [0, 0]
initial_conditions = [1 if i > 3 else 0 for i in x]

u = np.zeros((m,n))

u[0,:] = boundary_conditions[0]
u[-1,:] = boundary_conditions[1]
u[:,0] = initial_conditions

s = alpha**2 * (k/h**2)

A = np.diag([2+2*s]*(m-2),0) + np.diag([-s]*(m-3), -1) + np.diag([-s]*(m-3), 1)
B = np.diag([2-2*s]*(m-2),0) + np.diag([s]*(m-3), -1) + np.diag([s]*(m-3), 1)

for j in range(0, n-1):
    col = u[1:-1,j].copy()
    col = np.dot(B,col)
    col[0] = col[0] + s*(u[0,j] + u[0,j+1])
    col[-1] = col[-1] + s*(u[-1,j] + u[-1,j+1])
    u[1:-1,j+1] = solve(A,col)

xv, tv = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, tv, u.T, cmap=cm.coolwarm, rstride=1, cstride=1)
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'$u(x,t)$', rotation=90)
ax.set_title(r'Explicit Finite Difference Method, $h={}$, $k={}$'.format(h, k))
ax.view_init(elev=10, azim=-120)
ax.zaxis.set_rotate_label(False)
plt.show()