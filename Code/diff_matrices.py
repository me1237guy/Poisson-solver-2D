# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:11:50 2020

@author: Mohammad Asif Zaman

Differentiaion matrix operators in 1D and 2D
-   First derivatives d/dx, d/dy
-   Second derivatives d2/dx2, d2/dy2

Notes:
        - kron() is different in python compared to MATLAB. The matrix order is reveresed here.
          MATLAB version:
          Dx_2d = sp.kron(Dx_1d,Iy)
          Dy_2d = sp.kron(Ix,Dy_1d)

"""

import scipy.sparse as sp

def Diff_mat_1D(Nx):
    
    # First derivative
    D_1d = sp.diags([-1, 1], [-1, 1], shape = (Nx,Nx)) # A division by (2*dx) is required later.
    D_1d = sp.lil_matrix(D_1d)
    D_1d[0,[0,1,2]] = [-3, 4, -1]               # this is 2nd order forward difference (2*dx division is required)
    D_1d[Nx-1,[Nx-3, Nx-2, Nx-1]] = [1, -4, 3]  # this is 2nd order backward difference (2*dx division is required)
    
    # Second derivative
    D2_1d =  sp.diags([1, -2, 1], [-1,0,1], shape = (Nx, Nx)) # division by dx^2 required
    D2_1d = sp.lil_matrix(D2_1d)                  
    D2_1d[0,[0,1,2,3]] = [2, -5, 4, -1]                    # this is 2nd order forward difference. division by dx^2 required. 
    D2_1d[Nx-1,[Nx-4, Nx-3, Nx-2, Nx-1]] = [-1, 4, -5, 2]  # this is 2nd order backward difference. division by dx^2 required.
    
    return D_1d, D2_1d




def Diff_mat_2D(Nx,Ny):
    # 1D differentiation matrices
    Dx_1d, D2x_1d = Diff_mat_1D(Nx)
    Dy_1d, D2y_1d = Diff_mat_1D(Ny)


    # Sparse identity matrices
    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)


    
    # 2D matrix operators from 1D operators using kronecker product
    # First partial derivatives
    Dx_2d = sp.kron(Iy,Dx_1d)
    Dy_2d = sp.kron(Dy_1d,Ix)
    
    # Second partial derivatives
    D2x_2d = sp.kron(Iy,D2x_1d)
    D2y_2d = sp.kron(D2y_1d,Ix)
    
   
    
    # Return compressed Sparse Row format of the sparse matrices
    return Dx_2d.tocsr(), Dy_2d.tocsr(), D2x_2d.tocsr(), D2y_2d.tocsr()

def Diff_mat_3D(Nx, Ny, Nz):
    # Function to generate 1D differentiation matrices
    def Diff_mat_1D(N):
        D_1d = sp.diags([-1, 1], [-1, 1], shape=(N, N))
        D_1d = sp.lil_matrix(D_1d)
        D_1d[0, [0, 1, 2]] = [-3, 4, -1]
        D_1d[N - 1, [N - 3, N - 2, N - 1]] = [1, -4, 3]

        D2_1d = sp.diags([1, -2, 1], [-1, 0, 1], shape=(N, N))
        D2_1d = sp.lil_matrix(D2_1d)
        D2_1d[0, [0, 1, 2, 3]] = [2, -5, 4, -1]
        D2_1d[N - 1, [N - 4, N - 3, N - 2, N - 1]] = [-1, 4, -5, 2]

        return D_1d, D2_1d

    # Generate 1D differentiation matrices for x, y, and z directions
    Dx_1d, D2x_1d = Diff_mat_1D(Nx)
    Dy_1d, D2y_1d = Diff_mat_1D(Ny)
    Dz_1d, D2z_1d = Diff_mat_1D(Nz)

    # Sparse identity matrices for each direction
    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)
    Iz = sp.eye(Nz)

    # 3D matrix operators from 1D operators using kronecker product
    # First partial derivatives
    Dx_3d = sp.kron(sp.kron(Iz, Iy), Dx_1d)
    Dy_3d = sp.kron(sp.kron(Iz, Dy_1d), Ix)
    Dz_3d = sp.kron(sp.kron(Dz_1d, Iy), Ix)

    # Second partial derivatives
    D2x_3d = sp.kron(sp.kron(Iz, Iy), D2x_1d)
    D2y_3d = sp.kron(sp.kron(Iz, D2y_1d), Ix)
    D2z_3d = sp.kron(sp.kron(D2z_1d, Iy), Ix)

    # Return CSR format of the sparse matrices
    return Dx_3d.tocsr(), Dy_3d.tocsr(), Dz_3d.tocsr(), D2x_3d.tocsr(), D2y_3d.tocsr(), D2z_3d.tocsr()
