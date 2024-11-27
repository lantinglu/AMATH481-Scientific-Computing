import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import solve_ivp
from scipy.linalg import lu,solve_triangular
import time
from scipy.sparse.linalg import bicgstab, gmres

tspan = np.arange(0, 4.5, 0.5)
nu = 0.001
Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

#Define spatial domain and initial condition
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]

#Initial conditon
X, Y = np.meshgrid(x, y)
w0 = np.exp(-X ** 2 - Y ** 2 / 20).flatten()

kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

L = Lx #L = 20
n = N #n = 64 * 64
m = nx  #m = 64
delta = L / m #Distance

e0 = np.zeros((n, 1))
e1 = np.ones((n, 1))
e2 = np.copy(e1)
e4 = np.copy(e0)

#Matrix A
for j in range(1, m+1):
    e2[m*j-1] = 0
    e4[m*j-1] = 1

e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]

A = (spdiags(diagonals, offsets, n, n).toarray())/(delta**2)
A[0,0] = 2 / delta**2

#Matrix B
diagonals_B = [e1.flatten(), -e1.flatten(),e1.flatten(), -e1.flatten()]
offsets_B = [-(n-m), -m, m, (n-m)]
B = (spdiags(diagonals_B, offsets_B, n, n).toarray())/(2*delta)

#Matrix C
for i in range(1, n):
    e1[i] = e4[i - 1]

diagonals_C = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_C = [-m + 1, -1, 1,m - 1]
C = (spdiags(diagonals_C, offsets_C, n, n).toarray())/(2*delta)


#Part a
def spc_rhs(t, w0, nx, ny, N, A, B, C, K, nu):
    w = w0.reshape((nx,ny))
    wt = fft2(w)
    psit = -wt / K
    psi=np.real(ifft2(psit)).reshape(N)
    rhs=nu * np.dot(A, w0) + (np.dot(B, w0)) * (np.dot(C, psi)) - (np.dot(B, psi)) * (np.dot(C, w0))
    return rhs

wtsol = solve_ivp(spc_rhs, [0, 4], w0, t_eval = tspan, 
                  args = (nx, ny, N, A, B, C, K, nu), method = 'RK45')
A1 = wtsol.y
start_time = time.time()
end_time = time.time() 
elapsed_time = end_time - start_time
print('A1 = \n', A1)

#Plot A1
for j, t in enumerate(tspan):
    w = np.real(wtsol.y.T[j, :N].reshape((nx, ny)))
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w, shading='auto')
    plt.title(f'Time: {t}')
    plt.colorbar()

plt.tight_layout()
plt.show()

#Part b
def GE_rhs(t,w0, nx, ny, N, A, B, C, K, nu):
    psi = np.linalg.solve(A, w0)
    rhs = (nu * np.dot(A, w0) 
          + (np.dot(B, w0)) * (np.dot(C, psi)) 
          - (np.dot(B, psi)) * (np.dot(C, w0))
          )
    return rhs

P,L,U=lu(A)

def LU_rhs(t, w0, nx, ny, N, A, B, C, K, nu):
    Pb = np.dot(P,w0)
    y = solve_triangular(L,Pb,lower=True)
    psi = solve_triangular(U,y)
    rhs = (nu*np.dot(A, w0)
          + (np.dot(B, w0)) * (np.dot(C,psi)) 
          - (np.dot(B,psi)) * (np.dot(C,w0))
          )
    return rhs


wtsol = solve_ivp(GE_rhs, [0, 4], w0, t_eval=tspan, args=(nx, ny, N, A, B, C, K, nu), method='RK45')
A2 = wtsol.y
tart_time = time.time()
end_time = time.time()
elapsed_time = end_time - start_time
print('A2 = \n', A2)

#Plot A2
for j, t in enumerate(tspan):
    w = np.real(wtsol.y.T[j, :N].reshape((nx, ny)))
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w, shading='auto')
    plt.title(f'Time: {t}')
    plt.colorbar()
plt.tight_layout()
plt.show()

wtsol = solve_ivp(LU_rhs, [0, 4], w0, t_eval=tspan, args=(nx, ny, N, A, B, C, K, nu), method='RK45')
A3 = wtsol.y
start_time = time.time()
end_time = time.time()
elapsed_time = end_time - start_time
print('A3 = \n', A3)

#Plot A3
for j, t in enumerate(tspan):
    w = np.real(wtsol.y.T[j, :N].reshape((nx, ny)))
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, w, shading='auto')
    plt.title(f'Time: {t}')
    plt.colorbar()
plt.tight_layout()
plt.show()


#Part c
e1 = np.zeros((n, 1))
e1[::m] = 1

e2 = np.ones((n, 1))
e2[m-1::m] = 0

e3 = np.ones((n, 1))
e3[::m] = 0

e4 = np.zeros((n, 1))
e4[m-1::m] = 1

#diagonal elements
diagonals_C = [e1.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_C = [-m + 1, -1, 1, m - 1]

C = (spdiags(diagonals_C, offsets_C, n, n).toarray())/(2 * delta)

#Part d
import matplotlib.animation as animation
#domain
x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)

# Define initial conditions for different vortex configurations
initial_conditions = {
    "opposite_gaussians": lambda: np.exp(-((X - 5)**2 + Y**2)/20) - np.exp(-((X + 5)**2 + Y**2)/20),
    "same_gaussians": lambda: np.exp(-((X - 5)**2 + Y**2)/20) + np.exp(-((X + 5)**2 + Y**2)/20),
    "colliding_pairs": lambda: np.exp(-((X - 5)**2 + (Y - 5)**2)/20) - np.exp(-((X + 5)**2 + (Y + 5)**2)/20),
    "random_vortices": lambda: sum(((-1)**np.random.randint(0, 2)) * np.exp(-((X - np.random.uniform(-10, 10))**2 + (Y - np.random.uniform(-10, 10))**2)/20) for _ in range(10))
}

kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

def run_solver(initial_condition, label):
    w = initial_condition()
    wt = w.reshape(N)
    
    start_time = time.time()
    wtsol = solve_ivp(spc_rhs, [0, 4], wt, t_eval=tspan, args=(nx, ny, N, A, B, C, K, nu), method='RK45')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for {label}: {elapsed_time:.2f} seconds")

    #plot final state
    fig, ax = plt.subplots(figsize=(6, 6))
    def update_plot(i):
        ax.clear()
        state = wtsol.y[:, i].reshape((nx, ny))
        contour = ax.contourf(X, Y, state, levels=50, cmap='viridis')
        ax.set_title(f"Time: {tspan[i]:.2f} s")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        return contour

    ani = animation.FuncAnimation(fig, update_plot, frames=len(tspan), blit=False, repeat=False)
    plt.colorbar(ax.contourf(X, Y, w.reshape((nx, ny)), levels=50, cmap='viridis'))
    plt.title(f"Vortex Dynamics: {label}")
    ani.save(f'vortex_dynamics_{label}.gif', writer='ffmpeg', fps=10)
    plt.show()

for label, initial_condition in initial_conditions.items():
    run_solver(initial_condition, label)