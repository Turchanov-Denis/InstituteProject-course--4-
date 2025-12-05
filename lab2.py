import cmath
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# =====================================================
# 1. Бифуркационная диаграмма для параметра p
# =====================================================

def cubic_roots(p):
    """
    Решает кубическое уравнение
        y^3 + 3y^2 + (1+p)y + (p-1) = 0
    с помощью формулы Кардано.

    Метод:
    1. Приводим кубическое уравнение к форме y^3 + a*y^2 + b*y + c = 0.
    2. Выполняем замену y = x - a/3, чтобы убрать квадратный член (полином становится депрессированным).
    3. Вычисляем дискриминант D = (q/2)^2 + (p/3)^3.
    4. В зависимости от знака дискриминанта находим корни:
       - D > 0: один действительный и два комплексных.
       - D = 0: все корни действительные, хотя бы два совпадают.
       - D < 0: три разных действительных корня.
    """

    # коэффициенты исходного уравнения
    a = 3
    b = 1 + p
    c = p - 1

    # депрессированная форма: y = x - a/3
    a_third = a / 3
    p_dep = b - a * a_third
    q_dep = (2 * a * a * a) / 27 - (a * b) / 3 + c

    # дискриминант
    D = (q_dep / 2) ** 2 + (p_dep / 3) ** 3

    if D >= 0:
        # один действительный корень и два комплексных
        sqrt_D = cmath.sqrt(D)
        u = (-q_dep / 2 + sqrt_D) ** (1 / 3)
        v = (-q_dep / 2 - sqrt_D) ** (1 / 3)
        y1 = u + v - a_third
        y2 = -(u + v) / 2 - a_third + cmath.sqrt(3) * (u - v) / 2 * 1j
        y3 = -(u + v) / 2 - a_third - cmath.sqrt(3) * (u - v) / 2 * 1j
    else:
        # три действительных корня
        r = math.sqrt(-(p_dep / 3) ** 3)
        phi = math.acos(-q_dep / (2 * r))
        r_cbrt = (-p_dep / 3) ** 0.5
        y1 = 2 * r_cbrt * math.cos(phi / 3) - a_third
        y2 = 2 * r_cbrt * math.cos((phi + 2 * math.pi) / 3) - a_third
        y3 = 2 * r_cbrt * math.cos((phi + 4 * math.pi) / 3) - a_third

    return [y1, y2, y3]


# Диапазон параметра p
P = np.linspace(-2, 4, 400)

# Для каждого p собираем только действительные корни
roots_real = []
for pv in P:
    r = cubic_roots(pv)
    roots_real.append([root.real for root in r if np.isreal(root)])

plt.figure(figsize=(6, 4))

# Строим диаграмму: p на оси X, действительные корни на оси Y
for i, pv in enumerate(P):
    for r in roots_real[i]:
        plt.scatter(pv, r, color='blue', s=5)

# Отмечаем значение параметра, при котором происходит перестройка корней
plt.axvline(2, color='red', linestyle='--', label='точка бифуркации p* = 2')
plt.axhline(-1, color='green', linestyle='--', label='y* = -1')

plt.title("Бифуркационная диаграмма")
plt.xlabel("p")
plt.ylabel("y*")
plt.grid(True)
plt.legend()
plt.show()

# =====================================================
# 2. Фазовые траектории 3D-системы
# =====================================================

a = -1


def F(X):
    """
    Правая часть системы ОДУ:
        x' = a*x + y^2 + z^2
        y' = a*y + x^2 + z^2
        z' = a*z + x^2 + y^2

    На входе — вектор X = (x, y, z),
    на выходе — вектор производных.
    """
    x, y, z = X
    return np.array([a * x + y ** 2 + z ** 2,
                     a * y + x ** 2 + z ** 2,
                     a * z + x ** 2 + y ** 2])


def runge_kutta_5(F, X0, t_span, dt):
    """
    Краткое описание метода Рунге–Кутты:
    ------------------------------------
    1. Начальная точка: X0 в момент времени t0.
    2. На каждом шаге вычисляем несколько промежуточных наклонов k1...k6.
       Эти наклоны оценивают изменение X в разных точках текущего шага.
    3. Комбинируем наклоны, чтобы получить следующее приближённое значение X_{n+1}.
    4. Повторяем шаги до конца интервала [t0, t_end].
    """
    t0, t_end = t_span
    X = np.array(X0, dtype=float)
    t = t0

    t_values = [t]
    X_values = [X.copy()]

    while t < t_end:
        k1 = F(X)
        k2 = F(X + dt * k1 / 4)
        k3 = F(X + dt * (3 * k1 + 9 * k2) / 32)
        k4 = F(X + dt * (1932 * k1 - 7200 * k2 + 7296 * k3) / 2197)
        k5 = F(X + dt * (439 * k1 / 216 - 8 * k2 + 3680 * k3 / 513 - 845 * k4 / 4104))
        k6 = F(X + dt * (-8 * k1 / 27 + 2 * k2 - 3544 * k3 / 2565 + 1859 * k4 / 4104 - 11 * k5 / 40))

        # комбинация для 5-го порядка (Dormand-Prince)
        X = X + dt * (16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6)
        t += dt

        t_values.append(t)
        X_values.append(X.copy())

    return np.array(t_values), np.array(X_values).T  # транспонируем для совместимости с sol.y


# --- Визуализация ---
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Набор стартовых условий
initial_conditions = [
    (0.2, 0.3, 0.1),
    (1, 0.2, 0.1),
    (-0.5, -0.3, -0.2)
]

for X0 in initial_conditions:
    t_vals, X_vals = runge_kutta_5(F, X0, [0, 10], dt=0.01)
    ax.plot(X_vals[0], X_vals[1], X_vals[2])

# Отмечаем особые точки (стационарные решения)
special_points = [(0, 0, 0), (0.5, 0.5, 0.5)]
for pt in special_points:
    ax.scatter(*pt, s=50, label=f'Особая точка {pt}')

ax.set_title("Фазовые траектории 3D-системы")
ax.legend()
plt.show()


# =====================================================
# 3. Дискретная динамическая система
# =====================================================

def iterate(x0, N=30, limit=100):
    """
    Строим траекторию дискретной системы:
        x_{n+1} = -x_n + x_n^2.

    - Начинаем с X[0] = x0.
    - На каждом шаге вычисляем x_next по формуле.
    - Если значение растёт слишком сильно, прекращаем вычисления,
      чтобы избежать переполнения.
    """
    X = [x0]
    for _ in range(N):
        x_next = -X[-1] + X[-1] ** 2
        if abs(x_next) > limit:
            break
        X.append(x_next)
    return X

plt.figure(figsize=(6, 5))

for x0 in [-1, 3]:
    X = iterate(x0)
    plt.plot(X, label=f"x0={x0}")

plt.title("Итерации x_{n+1} = -x_n + x_n^2")
plt.xlabel("n")
plt.ylabel("x_n")
plt.grid(True)
plt.legend()
plt.show()
