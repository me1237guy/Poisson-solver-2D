from __future__ import print_function
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt  # Using matplotlib for plotting
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams.update({'font.size': 20})

from diff_matrices import Diff_mat_1D, Diff_mat_2D, Diff_mat_3D   
from utilities_definition import my_contourf_3d, my_scatter_3d, color_distinct
from boundary_definition import boundary_regions, boundary_regions_3d
from geo_source_definition import geo_src_def

# Call geometry + source function 
clr_set = color_distinct()   
x, y, z, ub_o, B_type_o, xb_i, yb_i, zb_i, ub_i, B_type_i, f =  geo_src_def('cap_3d')

Nx = len(x)
Ny = len(y)
Nz = len(z)

dx = x[1] - x[0]                
dy = y[1] - y[0]                
dz = z[1] - z[0]

X, Y, Z = np.meshgrid(x, y, z)
Xu = X.ravel()
Yu = Y.ravel()
Zu = Z.ravel()

Dx_3d, Dy_3d, Dz_3d, D2x_3d, D2y_3d, D2z_3d = Diff_mat_3D(Nx, Ny, Nz)

start_time = time.time()
B_ind, B_type, B_val = boundary_regions_3d(x, y, z, ub_o, B_type_o, xb_i, yb_i, zb_i, ub_i, B_type_i)
N_B = len(B_val)    
print("Boundary search time = %1.6s" % (time.time() - start_time))

plt.close('all')
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
my_scatter_3d(ax, Xu, Yu, Zu, clr_set[0], msize=4)

for m in range(N_B):
    if N_B - m < len(clr_set):
        color_index = N_B - m
    else:
        color_index = (N_B - m) % len(clr_set)
    
    my_scatter_3d(ax, Xu[B_ind[m]], Yu[B_ind[m]], Zu[B_ind[m]], clr_set[color_index], msize=4)

plt.figure(figsize=(9, 7))
my_contourf_3d(x, y, z, f.reshape(Nz, Ny, Nx), r'$f\,(x,y)$', 'RdBu')

start_time = time.time()
b = f  

for m in range(N_B):
    b[B_ind[m]] = B_val[m]

print("Right-hand vector construction time = %1.6s" % (time.time() - start_time))

I_sp = sp.eye(Nx * Ny * Nz).tocsr()  
L_sys = (D2x_3d / dx**2) + (D2y_3d / dy**2) + (D2z_3d / dz**2)

BD = I_sp  
BNx = Dx_3d  
BNy = Dy_3d  
BNz = Dz_3d  

start_time = time.time()

for m in range(N_B):
    if B_type[m] == 0:
        L_sys[B_ind[m], :] = BD[B_ind[m], :]
    elif B_type[m] == 1:
        L_sys[B_ind[m], :] = BNx[B_ind[m], :]
    elif B_type[m] == 2:
        L_sys[B_ind[m], :] = BNy[B_ind[m], :]
    elif B_type[m] == 3:
        L_sys[B_ind[m], :] = BNz[B_ind[m], :]

print("System matrix construction time = %1.6s" % (time.time() - start_time))

start_time = time.time()
u = spsolve(L_sys, b).reshape(Nz, Ny, Nx)  
print("spsolve() time = %1.6s" % (time.time() - start_time))

plt.figure(figsize=(12, 7))
plt.imshow(u[:, :, 0])  
plt.show()
