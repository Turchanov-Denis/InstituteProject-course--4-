"""
Лабораторная работа №3. Стохастические модели — метод Монте-Карло.
Вариант: область между двумя спиралями Архимеда (3 витка).
Интеграл: I = ∬_D (x^2 + y^3) dA.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------ 1. Параметры ------------------------

# Спирали
a1, b1 = 0.0, 1.0
a2, b2 = 0.0, 1.0
delta = 2 * np.pi
num_turns = 3

theta_max = 2 * np.pi * num_turns  # 0 → 6π

# Макс радиус
r_max = a2 + b2 * theta_max + delta     # = 8π

# Охватывающий прямоугольник
xmin, xmax = -r_max, r_max
ymin, ymax = -r_max, r_max
S_box = (xmax - xmin) * (ymax - ymin)

# Параметры эксперимента
Ns = [1_000, 3_000, 10_000, 30_000, 100_000]
SEED = 42


# ------------------------ 2. Спирали ------------------------

def r_in(theta):
    return a1 + b1 * theta      # θ

def r_out(theta):
    return a2 + b2 * theta + delta  # θ + 2π


def f_xy(x, y):
    """Подынтегральная функция"""
    return x**2 + y**3


# ------------------------ 3. Принадлежность точек области ------------------------

def region_mask(x, y):
    """
    Определяет, какие точки (x, y) принадлежат области D между
    двумя спиралями Архимеда.

    Идея:
      1) Переходим к полярным координатам: r = sqrt(x^2 + y^2), θ0 = atan2(y, x).
      2) Функции спиралей заданы для θ ∈ [0, θ_max].
         Но atan2 возвращает угол в диапазоне (-π, π], поэтому
         сначала приводим θ0 к [0, 2π), затем добавляем 2π * k
         (k = 0, 1, ..., num_turns-1), чтобы получить возможные углы θ_k
         в [0, θ_max].
      3) Точка принадлежит области, если существует такое θ_k, что
            r_in(θ_k) <= r <= r_out(θ_k).
    """

    r = np.hypot(x, y)
    theta0 = np.arctan2(y, x)

    # приводим угол в [0,2π)
    theta0 = np.where(theta0 < 0, theta0 + 2*np.pi, theta0)

    inside = np.zeros_like(r, dtype=bool)

    # перебираем витки
    for k in range(num_turns+1):
        theta = theta0 + 2*np.pi*k
        valid = (theta >= 0) & (theta <= theta_max)

        rin = r_in(theta)
        rout = r_out(theta)

        inside |= valid & (r >= rin) & (r <= rout)

    return inside


# ------------------------ 4. Метод Монте-Карло ------------------------

def monte_carlo_spirals(N, rng):
    """
    Оценка площади области D и интеграла I = ∬_D (x^2 + y^3) dA
    методом Монте-Карло.

    Алгоритм:
      1) Равномерно генерируем N точек (x_i, y_i) в охватывающем
         прямоугольнике B: [xmin, xmax] × [ymin, ymax].
      2) Для каждой точки определяем, попала ли она в область D:
            inside_i ∈ {0, 1}.
      3) Оценка площади:
            p̂ = (1/N) * Σ inside_i      — доля попаданий в D;
            Ŝ(D) = S_box * p̂          — оценка площади области.
         Стандартная ошибка (SE) площади:
            SE_S = S_box * sqrt( p̂(1 - p̂) / N ).
      4) Оценка интеграла:
            I = ∬_D f(x, y) dA ≈ S_box * (1/N) * Σ [ f(x_i, y_i) * inside_i ].
         Стандартная ошибка (SE) интеграла:
            оцениваем дисперсию по выборке и используем σ̂ / sqrt(N).
    """

    # равномерные
    xs = rng.uniform(xmin, xmax, size=N)
    ys = rng.uniform(ymin, ymax, size=N)

    inside = region_mask(xs, ys)

    # площадь
    p_hat = inside.mean() # доля точек в области
    area_est = S_box * p_hat
    area_se = S_box * np.sqrt(p_hat*(1-p_hat)/N)

    # интеграл
    values = f_xy(xs, ys) * inside
    integral_est = S_box * values.mean()
    # Выборочная дисперсия (ddof=1) для оценки стандартной ошибки
    var_hat = values.var(ddof=1) if N>1 else 0
    integral_se = S_box * np.sqrt(var_hat/N)

    return area_est, area_se, integral_est, integral_se


# ------------------------ 5. MAIN ------------------------

rng = np.random.default_rng(SEED)

rows = []
for N in Ns:
    A, Ase, I, Ise = monte_carlo_spirals(N, rng)
    rows.append((N, A, Ase, I, Ise))

df = pd.DataFrame(rows, columns=["N","Area","Area_SE","Integral","Int_SE"])
print(df)

# ------------------------ 8. 5 Рисунков с точками внутри области ------------------------

Ns_plot = [1_000, 5_000, 10_000, 50_000, 100_000]

plt.figure(figsize=(12,10))

for i, N in enumerate(Ns_plot, start=1):
    xs = rng.uniform(xmin, xmax, size=N)
    ys = rng.uniform(ymin, ymax, size=N)
    inside = region_mask(xs, ys)

    plt.subplot(3, 2, i)      # сетка 3 × 2
    plt.scatter(xs[inside], ys[inside], s=1, color='blue')
    plt.title(f"N = {N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal", "box")

plt.tight_layout()
plt.show()

# ------------------------ 6. График сходимости площади ------------------------

plt.figure()
plt.plot(df["N"], df["Area"], marker='o')
plt.fill_between(df["N"],
                 df["Area"] - 2*df["Area_SE"],
                 df["Area"] + 2*df["Area_SE"],
                 alpha=0.2)
plt.xscale("log")
plt.xlabel("Количество точек N (log)")
plt.ylabel("Площадь области")
plt.title("Сходимость площади (Монте-Карло)")
plt.grid(True)
plt.show()


# ------------------------ 7. График сходимости интеграла ------------------------

plt.figure()
plt.plot(df["N"], df["Integral"], marker='o')
plt.fill_between(df["N"],
                 df["Integral"] - 2*df["Int_SE"],
                 df["Integral"] + 2*df["Int_SE"],
                 alpha=0.2)
plt.xscale("log")
plt.xlabel("Количество точек N (log)")
plt.ylabel("Интеграл")
plt.title("Сходимость интеграла (Монте-Карло)")
plt.grid(True)
plt.show()


