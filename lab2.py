import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Бифуркационная диаграмма
# ---------------------------

def cubic_roots(p):
    # Корни уравнения y^3 + 3y^2 + (1+p)y + (p-1) = 0
    coeffs = [1, 3, 1+p, p-1]
    return np.roots(coeffs)

P = np.linspace(-2, 4, 400)
roots_real = []

for pv in P:
    r = cubic_roots(pv)
    roots_real.append([root.real for root in r if np.isreal(root)])

plt.figure(figsize=(6,4))
for i, pv in enumerate(P):
    for r in roots_real[i]:
        plt.scatter(pv, r, color='blue', s=5)
plt.axvline(2, color='red', linestyle='--', label='точка бифуркации p* = 2')
plt.axhline(-1, color='green', linestyle='--', label='y* = -1')
plt.title("Бифуркационная диаграмма")
plt.xlabel("p")
plt.ylabel("y*")
plt.grid(True)
plt.legend()
plt.show()


# ---------------------------
# 2. Фазовые траектории 3D-системы
# ---------------------------

a = -1

def F(X):
    x, y, z = X
    return [a*x + y**2 + z**2,
            a*y + x**2 + z**2,
            a*z + x**2 + y**2]

from scipy.integrate import solve_ivp

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

initial_conditions = [(0.2,0.3,0.1), (1,0.2,0.1), (-0.5,-0.3,-0.2)]

for X0 in initial_conditions:
    sol = solve_ivp(lambda t, X: F(X), [0, 10], X0, max_step=0.01)
    ax.plot(sol.y[0], sol.y[1], sol.y[2])

special_points = [(0,0,0), (0.5,0.5,0.5)]
for pt in special_points:
    ax.scatter(*pt, s=50, label=f'Особая точка {pt}')
ax.set_title("Фазовые траектории 3D-системы")
ax.legend()
plt.show()


# ---------------------------
# 3. Дискретная система
# ---------------------------

def iterate(x0, N=30, limit=50):
    X = [x0]
    for _ in range(N):
        x_next = -X[-1] + X[-1]**2
        if abs(x_next) > limit:  # Останавливаем рост
            break
        X.append(x_next)
    return X

plt.figure(figsize=(6,4))
for x0 in [-1, 3]:
    X = iterate(x0)
    plt.plot(X, label=f"x0={x0}")


plt.title("Итерации x_{n+1} = -x_n + x_n^2")
plt.xlabel("n")
plt.ylabel("x_n")
plt.grid(True)
plt.legend()
plt.show()
