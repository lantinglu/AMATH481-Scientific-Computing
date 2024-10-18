import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def shoot_eq(phi, x, epsilon):
    return [phi[1], (x**2 - epsilon) * phi[0]]

#initialize paramters
tol = 1e-4
L = 4
dx = 0.1
colors = ['r', 'b', 'g', 'c', 'm']
x_values = np.arange(-L, L + dx, dx)
epsilon_start = 0.1

A1 = np.zeros((len(x_values), 5))
A2 = np.zeros(5)

for modes in range(1, 6):
    epsilon = epsilon_start
    depsilon = 0.2
    for _ in range(1000):
        y0 = [1, np.sqrt(L**2 - epsilon)]
        y = odeint(shoot_eq, y0, x_values, args=(epsilon, ))

        #check tol
        if abs(y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0]) < tol:
            break

        if ((-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(L**2 - epsilon) * y[-1, 0])) > 0:
            epsilon += depsilon
        else:
            epsilon -= depsilon
            depsilon /= 2
    A2[modes - 1] = epsilon
    epsilon_start = epsilon + 0.1
    norm = np.trapz(y[:, 0] * y[:, 0], x_values)
    eigenfuction = abs(y[:, 0] / np.sqrt(norm))
    A1[:, modes - 1] = eigenfuction
    plt.plot(x_values, eigenfuction, colors[modes - 1])

print('A1 = \n', A1)
print('A2 = \n', A2)

plt.title('Eigenfunctions for Different Modes')
plt.xlabel('x')
plt.ylabel('Eigenfunction')
plt.legend()
plt.grid(True)
plt.show()