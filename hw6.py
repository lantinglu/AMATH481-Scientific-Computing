import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from numpy import *
from scipy.linalg import kron

#Define parameters
tspan = np.linspace(0, 4, 9)
tshoot = [0, 4]
D1, D2 = 0.1, 0.1
m = 1
beta = 1
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

# Define spatial domain and initial conditions
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

# Define spectral k values
kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

#Compute U and V
U = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(X + 1j * Y) - np.sqrt(X**2 + Y**2))
V = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + 1j * Y) - np.sqrt(X**2 + Y**2))

# Define the ODE system
def spc_rhs(t, UVt2, nx, ny, N, K, beta, D1, D2):
    Utc = UVt2[0: N]
    Vtc = UVt2[N: 2*N]
    Ut = Utc.reshape((nx, ny)) 
    Vt = Vtc.reshape((nx, ny))
    Uo = ifft2(Ut)
    Vo = ifft2(Vt)
    rhs_u = (fft2(Uo - Uo**3 - Vo**2 * Uo + beta * Uo**2 * Vo + beta * Vo**3) - D1 * K * Ut).reshape(N)
    rhs_v = (fft2(-beta * Uo**3 - beta * Vo**2 * Uo + Vo - Uo**2 * Vo - Vo**3) - D2 * K * Vt).reshape(N)
    rhs = np.hstack([rhs_u, rhs_v])
    return rhs

# Solve the ODE and plot the results
wt0= np.hstack([fft2(U).reshape(N), fft2(V).reshape(N)])
wtsol = solve_ivp(spc_rhs, tshoot, wt0, t_eval = tspan, args=(nx, ny, N, K, beta, D1, D2), method = 'RK45')
A1 = wtsol.y
print('A1 = \n', A1)

for j, t in enumerate(tspan):
    w = np.real(ifft2(wtsol.y[N:,j].reshape((nx, ny))))
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w)
    plt.title(f'Time: {t}')
    plt.colorbar()

plt.tight_layout()
plt.show()

def cheb(N):
    if N == 0: 
        D = 0.; x = 1.
    else:
        n = arange(0, N + 1)
        x = cos(pi * n / N).reshape(N + 1, 1) 
        c = (hstack(( [2.], ones(N - 1), [2.])) * (-1) ** n).reshape(N + 1, 1)
        X = tile(x,(1, N + 1))
        dX = X - X.T
        D = dot(c,1./c.T)/(dX+eye(N+1))
        D -= diag(sum(D.T,axis=0))
    return D, x.reshape(N+1)

N = 30
D, x_value = cheb(N)
D[N, :] = 0
D[0, :] = 0
D = 0.1 * D
Dxx = np.dot(D, D)

x = x_value * 10
y = x
I = np.eye(len(Dxx))
L = np.kron(I, Dxx) + np.kron(Dxx, I)
X, Y = np.meshgrid(x, y)

nx, ny = N + 1, N + 1
N2 = nx * ny

m = 1
beta = 1

U = np.tanh(np.sqrt(X**2 + Y**2)) * np.cos(m * np.angle(X + 1j * Y) - np.sqrt(X**2 + Y**2))
V = np.tanh(np.sqrt(X**2 + Y**2)) * np.sin(m * np.angle(X + 1j * Y) - np.sqrt(X**2 + Y**2))

def spc_rhs2(t, UVt2, N2, L, beta, D1, D2):
    Utc = UVt2[0:N2]
    Vtc = UVt2[N2:2 * N2]
    rhs_u = (Utc - Utc**3 - Vtc**2 * Utc + beta * Utc**2 * Vtc + beta * Vtc**3 + D1 * np.dot(L, Utc)).reshape((nx, ny))
    rhs_v = (-beta * Utc**3 - beta * Vtc**2 * Utc + Vtc - Utc**2 * Vtc - Vtc**3 + D2 * np.dot(L, Vtc)).reshape((nx, ny))
    return np.hstack([rhs_u.ravel(), rhs_v.ravel()])

wt0 = np.hstack([U.ravel(), V.ravel()])
wtsol = solve_ivp(spc_rhs2, tshoot, wt0, t_eval=tspan, args=(N2, L, beta, D1, D2), method='RK45')
A2 = wtsol.y
print('A2 = \n', A2)

for j, t in enumerate(tspan):
    w = wtsol.y[:N2, j].reshape((nx, ny))
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w)
    plt.title(f'Time: {t}')
    plt.colorbar()

plt.tight_layout()
plt.show()