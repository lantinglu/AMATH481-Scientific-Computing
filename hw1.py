import numpy as np
#Problem 1
#i
def f(x):
    return x * np.sin(3 * x) - np.exp(x)

def df(x):
    return np.sin(3 * x) + 3 * x * np.cos(3 * x) - np.exp(x)

def newton_raphson(x0, tol=1e-6):
    A1 = [x0]
    x = x0
    for i in range(1000):
        fx = f(x)
        dfx = df(x)
        x1 = x - fx / dfx
        A1.append(x1)
        x = x1
        if abs(fx) < tol:
            break
    iterations = len(A1) - 1    
    return A1, iterations

#ii
def bisection(a, b, tol=1e-6):
    if f(a) * f(b) >= 0:
        return None, None
    
    A2 = []
    iterations = 0
    for i in range(2000):
        midpoint = (a + b) / 2.0
        A2.append(midpoint)
        fmid = f(midpoint)
        if fmid > 0:
            a = midpoint
        else:
            b = midpoint
        iterations += 1
        if abs(fmid) < tol:
            break
    return A2, iterations

A1, newton_iterations = newton_raphson(-1.6)
A2, bisection_iterations = bisection(-0.7, -0.4)
A3 = [newton_iterations, bisection_iterations]

print("A1 = ", A1)
print("A2 = ", A2)
print("A3 = ", A3)

#Problem 2
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

A4 = A + B
A5 = (3 * x - 4 * y).flatten()
A6 = (A @ x).flatten()
A7 = (B @ (x - y)).flatten()
A8 = (D @ x).flatten()
A9 = (D @ y + z).flatten()
A10 = A @ B
A11 = B @ C
A12 = C @ D


print("A4 = ", A4)
print("A5 = ", A5)
print("A6 = ", A6)
print("A7 = ", A7)
print("A8 = ", A8)
print("A9 = ", A9)
print("A10 = ", A10)
print("A11 = ", A11)
print("A12 = ", A12)
