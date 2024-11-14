import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

n = 8
x = np.linspace(-10, 10, n+1)
x = x[:n]
dx = x[1] - x[0]
a = n * n #size of the matrix

e0 = np.zeros((a, 1))
e1 = np.ones((a, 1))
e2 = np.copy(e1)
e4 = np.copy(e0)

for j in range(1, n+1):
    e2[n*j-1] = 0 
    e4[n*j-1] = 1

e3 = np.zeros_like(e2)
e3[1:a] = e2[0:a-1]
e3[0] = e2[a-1]

e5 = np.zeros_like(e4)
e5[1:a] = e4[0:a-1]
e5[0] = e4[a-1]

diagonals_A = [e1.flatten(), e1.flatten(), e5.flatten(),
             e2.flatten(), -4 * e1.flatten(), e3.flatten(),
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets_A = [-(a-n), -n, -n+1, -1, 0, 1, n-1, n, (a-n)]
matrixA = spdiags(diagonals_A, offsets_A, a, a).toarray() / dx**2
A1 = matrixA

diagonals_B = [e1.flatten(), -e1.flatten(), e1.flatten(), -e1.flatten()]
offsets_B = [-(a-n), -n, n, (a-n)]
matrixB = spdiags(diagonals_B, offsets_B, a, a).toarray() / (2*dx)
A2 = matrixB

diagonals_C = [e5.flatten(), -e2.flatten(), e3.flatten(), -e4.flatten()]
offsets_C = [-n+1, -1, 1, n-1]
matrixC = spdiags(diagonals_C, offsets_C, a, a).toarray() / (2*dx)
A3 = matrixC

print('A1 = \n', A1)
print('A2 = \n', A2)
print('A3 = \n', A3)

#plot
plt.spy(matrixA)
plt.title('Matrix Structure_A')
plt.show()

plt.spy(matrixB)
plt.title('Matrix Structure_B')
plt.show()

plt.spy(matrixC)
plt.title('Matrix Structure_C')
plt.show()