import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.integrate import solve_bvp, simpson, solve_ivp
import math


#Part a
def shoot_eq(x, phi, epsilon):
    return [phi[1], (x**2 - epsilon) * phi[0]]

#initialize paramters
tol = 1e-4
L = 4
dx = 0.1
colors = ['r', 'b', 'g', 'c', 'm']
x_values = np.arange(-L, L + dx, dx)
epsilon_start = 0.1
eigenvalus = []
eigenfunction = []

A1 = np.zeros((len(x_values), 5))
A2 = np.zeros(5)

for modes in range(1, 6):
    epsilon = epsilon_start
    depsilon = 0.2
    for _ in range(1000):
        y0 = [1, np.sqrt(L**2 - epsilon)]
        res = solve_ivp(shoot_eq,
                        [x_values[0], x_values[-1]],
                        y0,
                        args=(epsilon,),
                        t_eval=x_values
                        )
        if abs(res.y[1, -1] + np.sqrt(L**2 - epsilon) * res.y[0, -1]) < tol:
            break
        if ((-1) ** (modes + 1) * (res.y[1, -1] + np.sqrt(L**2 - epsilon) * res.y[0, -1])) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon
            depsilon /= 2

    A2[modes - 1] = epsilon
    epsilon_start = epsilon + 0.1
    norm = np.trapz(res.y[0] * res.y[0], x_values)
    eigenfuction = abs(res.y[0] / np.sqrt(norm))
    A1[:, modes - 1] = eigenfuction
    plt.plot(x_values, eigenfuction, colors[modes - 1])

print('A1 = \n', A1)
print('A2 = \n', A2)


# Part b
L = 4
dx = 0.1
x = np.arange(-L, L + dx, dx)
N = len(x) - 2

M = np.zeros((N, N))
for i in range(N):
    M[i, i] = -2 - (x[i + 1] ** 2) * (dx ** 2)

for i in range(N - 1):
    M[i, i + 1] = 1
    M[i + 1, i] = 1

M1 = M

M2 = np.zeros((N, N))
M2[0, 0] = 4 / 3
M2[0, 1] = -1 / 3

M3 = np.zeros((N, N))
M3[N - 1, N - 2] = -1 / 3
M3[N - 1, N - 1] = 4 / 3

M = M1 + M2 + M3
M = M / (dx ** 2)

D, V = eigs(- M, k=5, which='SM')

# boundry conditions
phi_0 = (4 / 3) * V[0, :] - (1 / 3) * V[1, :]
phi_n = - (1 / 3) * V[-2, :] + (4 / 3) * V[-1, :]

V = np.vstack((phi_0, V, phi_n))

# normalize
for i in range(5):
    norm = np.trapz(V[:, i] ** 2, x)
    V[:, i] = abs(V[:, i] / np.sqrt(norm))
    plt.plot(x, V[:, i])

plt.legend(["$\\phi_1$", "$\\phi_2$", "$\\phi_3$", "$\\phi_4$", "$\\phi_5$"], loc="upper right")
plt.xlabel("x")
plt.ylabel("Eigenfunctions")
plt.title("First Five Normalized Eigenfunctions")
plt.grid()

A3 = V
A4 = D

print('A3 = \n', A3)
print('A4 = \n', A4)

plt.show()

#Part c
L = 2
K = 1
dx = 0.1
tol = 1e-6
xshoot = np.arange (-L, L + dx, dx)
gamma_values = [0.05, - 0.05]

A5, A7 = np.zeros((len (xshoot), 2)), np.zeros((len (xshoot), 2))
A6, A8 = np.zeros(2), np.zeros (2)


def shoot2(x, phi, epsilon, gamma):
    return [phi[1], 
            (gamma * abs(phi[0])**2 + K * x**2 - epsilon) * phi[0]]

for gamma in gamma_values:
    epsilon_start = -1
    A = 1e-6
    for modes in range(1, 3):
        dA = 0.01
        for k in range (100):
            epsilon = epsilon_start
            depsilon = 0.2
            for kk in range (100):
                phi0 = [A, np.sqrt (K * L ** 2 - epsilon) * A]
                ans = solve_ivp(
                    lambda x, phi: shoot2(x, phi, epsilon, gamma),
                    [xshoot[0], xshoot[-1]],
                    phi0,
                    t_eval=xshoot
                    )
                phi_sol = ans.y.T
                x_sol = ans.t
                bc = phi_sol[-1, 1] + np.sqrt (L ** 2 - epsilon) * phi_sol[-1, 0]
                if abs (bc) < tol:
                    break
                if (-1) ** (modes + 1) * bc > 0:
                    epsilon += depsilon
                else:
                    epsilon -= depsilon
                    depsilon /= 2

            #check if it is focused
            integral = simpson (phi_sol[:, 0] ** 2, x=x_sol)
            if abs (integral - 1) < tol:
                break
            if integral < 1:
                A += dA
            else:
                A -= dA
                dA /= 2

        epsilon_start = epsilon + 0.2

        if gamma > 0:
            A5[:, modes - 1] = np.abs (phi_sol[:, 0])
            A6[modes - 1] = epsilon

        else:
            A7[:, modes - 1] = np.abs (phi_sol[:, 0])
            A8[modes - 1] = epsilon

plt.plot (xshoot, A5)
plt.plot (xshoot, A7)
plt.legend (["$\\phi_1$", "$\\phi_2$"], loc="upper right")
plt.show ()

print('A5 = \n', A5)
print('A6 = \n', A6)
print('A7 = \n', A7)
print('A8 = \n', A8)


#Part d
def rhs_a(x, phi, epsilon):
    return [phi[1], (x ** 2 - epsilon) * phi[0]]

L = 2
x_span = [-L, L]
epsilon = 1
A = 1
phi0 = [A, np.sqrt(L ** 2 - epsilon) * A]
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

dt45, dt23, dtRadua, dtBDF = [], [], [], []

for tol in tols:
    options = {'rtol': tol, 'atol': tol}
    sol45 = solve_ivp(rhs_a, x_span, phi0, method='RK45', args=(epsilon,), **options)
    sol23 = solve_ivp(rhs_a, x_span, phi0, method='RK23', args=(epsilon,), **options)
    solRadua = solve_ivp(rhs_a, x_span, phi0, method='Radau', args=(epsilon,), **options)
    solBDF = solve_ivp(rhs_a, x_span, phi0, method='BDF', args=(epsilon,), **options)

    dt45.append(np.mean(np.diff(sol45.t)))
    dt23.append(np.mean(np.diff(sol23.t)))
    dtRadua.append(np.mean(np.diff(solRadua.t)))
    dtBDF.append(np.mean(np.diff(solBDF.t)))

fit45 = np.polyfit(np.log(dt45), np.log(tols), 1)
fit23 = np.polyfit(np.log(dt23), np.log(tols), 1)
fitRadua = np.polyfit(np.log(dtRadua), np.log(tols), 1)
fitBDF = np.polyfit(np.log(dtBDF), np.log(tols), 1)

#extract slope
slope45 = fit45[0]
slope23 = fit23[0]
slopeRadua = fitRadua[0]
slopeBDF = fitBDF[0]

A9 = np.array([slope45, slope23, slopeRadua, slopeBDF])

print('A9 = \n', A9)


#Part e
def H0(x):
    return np.ones_like(x)
def H1(x):
    return 2 * x
def H2(x):
    return 4 * (x ** 2) - 2
def H3(x):
    return 8 * (x ** 3) - 12 * x
def H4(x):
    return 16 * (x ** 4) - 48 * (x ** 2) + 12

L = 4
dx = 0.1
x = np.arange(-L, L + dx, dx)

h = np.column_stack([H0(x), H1(x), H2(x), H3(x), H4(x)])
phi = np.zeros(h.shape)

#normalize and append
for i in range(5):
    phi[:, i] = ((np.exp(- (x ** 2) / 2) * h[:, i]) /
                 np.sqrt(math.factorial(i) * (2 ** i) * np.sqrt(np.pi))
                 )

eigen_funa = np.zeros(5)
eigen_funb = np.zeros(5)
eigen_vala = np.zeros(5)
eigen_valb = np.zeros(5)

for i in range(5):
    eigen_funa[i] = simpson(((abs(A1[:, i])) - abs(phi[:, i])) ** 2, x=x)
    eigen_funb[i] = simpson(((abs(A3[:, i])) - abs(phi[:, i])) ** 2, x=x)

    eigen_vala[i] = 100 * (abs(A2[i] - (2 * (i + 1) - 1)) / (2 * (i + 1) - 1))
    eigen_valb[i] = 100 * (abs(A4[i] - (2 * (i + 1) - 1)) / (2 * (i + 1) - 1))

A10 = eigen_funa
A12 = eigen_funb
A11 = eigen_vala
A13 = eigen_valb

print('A10 = \n', A10)
print('A11 = \n', A11)
print('A12 = \n', A12)
print('A13 = \n', A13)
