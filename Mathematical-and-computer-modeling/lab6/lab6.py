"""
Лабораторная работа №6. «Модель диффузии»
Вариант 26

Решение уравнения:
    U_t = D(x) U_xx - f(x) U + 5

Коэффициенты:
    D(x) = x + 1
    f(x) = x + 1

Область:
    0 ≤ x ≤ 10
    0 ≤ t ≤ T   (в работе T = 1)

Начальное условие:
    U(0,x) = x^2 (10 - x)

Граничные условия:
    U(t,0) = 0
    U(t,10) = 0

Цели работы:
1) Построить явную и неявную разностные схемы.
2) Исследовать порядок аппроксимации.
3) Сравнить точность и устойчивость схем.
4) Выполнить численные эксперименты и построить графики.

ВНИМАНИЕ:
Код полностью переработан в стиле методического примера.
Все комментарии — максимально подробные.
"""

# ========= ИМПОРТ БИБЛИОТЕК =========
import numpy as np
import matplotlib.pyplot as plt
from math import log2, sqrt
import os


# ========= ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ =========

L = 10.0     # длина интервала по x
T = 1.0      # конечное время моделирования


# ========= КОЭФФИЦИЕНТЫ И УСЛОВИЯ =========

def D_coef(x):
    """Коэффициент диффузии: D(x) = x + 1."""
    return x + 1.0


def f_coef(x):
    """Коэффициент при U в реакционном члене: f(x) = x + 1."""
    return x + 1.0


def u_init(x):
    """Начальное условие: U(0,x) = x^2 (10 - x)."""
    return x**2 * (10.0 - x)


def u_left(t):
    """Левое граничное условие: U(t,0) = 0."""
    return 0.0


def u_right(t):
    """Правое граничное условие: U(t,10) = 0."""
    return 0.0


def rmse(a, b):
    """
    Среднеквадратичная ошибка (Root Mean Square Error).
    RMSE = sqrt( mean((a - b)^2) )
    """
    return float(sqrt(np.mean((a - b)**2)))


# ========= 1. ЯВНАЯ СХЕМА (FTCS) =========

def solve_explicit(Nx, Nt):
    """
    Явная разностная схема (FTCS — Forward Time, Central Space):

        U_i^{n+1} = U_i^n +
            τ * [ D_i * (U_{i+1}^n - 2U_i^n + U_{i-1}^n)/h²  - f_i U_i^n + 5 ]

    Схема имеет порядок:
        O(τ) по времени, O(h²) по пространству.

    Условие устойчивости:
        τ ≤ h² / (2 * max D)
        В нашей задаче max D = D(10) = 11.
    """
    h = L / Nx
    tau = T / Nt

    # Узлы сетки
    x = np.linspace(0.0, L, Nx + 1)
    t = np.linspace(0.0, T, Nt + 1)

    D_vals = D_coef(x)
    f_vals = f_coef(x)

    # Матрица решения
    U = np.zeros((Nt + 1, Nx + 1))

    # Начальное и граничные условия
    U[0, :] = u_init(x)
    U[0, 0] = 0; U[0, -1] = 0

    # Итерации по времени
    for n in range(Nt):
        # Границы на новом шаге
        U[n + 1, 0] = 0
        U[n + 1, -1] = 0

        # Внутренние узлы
        for i in range(1, Nx):
            U_xx = (U[n, i + 1] - 2*U[n, i] + U[n, i - 1]) / h**2
            rhs = D_vals[i] * U_xx - f_vals[i] * U[n, i] + 5.0
            U[n + 1, i] = U[n, i] + tau * rhs

    return x, t, U


# ========= 2. НЕЯВНАЯ СХЕМА (BACKWARD EULER) =========

def solve_implicit(Nx, Nt):
    """
    Неявная схема (Backward Euler):
        (1 + 2α_i + τ f_i)*U_i^{n+1}
        - α_i U_{i-1}^{n+1}
        - α_i U_{i+1}^{n+1}
        = U_i^n + τ*5

    где α_i = τ D_i / h².

    На каждом шаге решаем трёхдиагональную СЛАУ методом прогонки.
    Схема безусловно устойчива.
    """
    h = L / Nx
    tau = T / Nt

    x = np.linspace(0.0, L, Nx + 1)
    t = np.linspace(0.0, T, Nt + 1)

    D_vals = D_coef(x)
    f_vals = f_coef(x)

    U = np.zeros((Nt + 1, Nx + 1))
    U[0, :] = u_init(x)

    # Размерность внутренней части (i = 1..Nx-1)
    Nint = Nx - 1

    # Диагонали для прогонки
    a = np.zeros(Nint)
    b = np.zeros(Nint)
    c = np.zeros(Nint)
    d = np.zeros(Nint)

    for n in range(Nt):
        U[n + 1, 0] = 0; U[n + 1, -1] = 0

        for k in range(Nint):
            i = k + 1
            alpha = tau * D_vals[i] / h**2

            a[k] = -alpha
            b[k] = 1 + 2*alpha + tau * f_vals[i]
            c[k] = -alpha
            d[k] = U[n, i] + tau * 5.0

        # Метод прогонки
        for k in range(1, Nint):
            w = a[k] / b[k - 1]
            b[k] -= w * c[k - 1]
            d[k] -= w * d[k - 1]

        y = np.zeros(Nint)
        y[-1] = d[-1] / b[-1]
        for k in range(Nint - 2, -1, -1):
            y[k] = (d[k] - c[k] * y[k + 1]) / b[k]

        U[n + 1, 1:Nx] = y

    return x, t, U


# ========= 3. ИССЛЕДОВАНИЕ ПОРЯДКА СХЕМЫ =========

def convergence_experiment():
    """
    Исследование порядка аппроксимации.
    Используем RMSE + log2(e_h / e_{h/2}).

    Подход:
    1. На самой мелкой сетке (Nx_max = 160) считаем "почти точное" решение.
    2. На более грубых сетках сравниваем решение с эталоном.
    3. Ошибка: RMSE.
    4. Порядок: p = log2(err(h) / err(h/2))
    """
    Nx_list = [20, 40, 80, 160]
    C = 5  # Nt = C * Nx² (ρ = τ/h² постоянна => порядок видно хорошo)

    results = []

    Nx_fine = Nx_list[-1]
    Nt_fine = int(C * Nx_fine * Nx_fine)
    x_f, t_f, U_f = solve_implicit(Nx_fine, Nt_fine)
    U_f_T = U_f[-1, :]

    print("\n=== Исследование порядка аппроксимации ===")
    print(" Nx    Nt        RMSE")

    for Nx in Nx_list[:-1]:
        Nt = int(C * Nx * Nx)
        x, t, U = solve_implicit(Nx, Nt)
        U_T = U[-1, :]

        # Интерполяция эталонного решения на грубую сетку
        U_ref = np.interp(x, x_f, U_f_T)

        err = rmse(U_T, U_ref)
        results.append(err)

        print(f"{Nx:3d}  {Nt:6d}   {err:10.6e}")

    print("\nОценка порядка p = log₂(e_h / e_{h/2}):")
    for k in range(1, len(results)):
        e1 = results[k - 1]
        e2 = results[k]
        p = log2(e1 / e2)
        print(f"  h{k} → h{k+1}: p ≈ {p:.3f}")



def plot_slices(x, t, U_exp, U_imp, times=(0.0, 0.25, 0.5, 1.0)):
    """
    Сравнение профилей U(x,t) для явной и неявной схем.
    """
    plt.figure(figsize=(8, 6))
    for tt in times:
        n = np.argmin(abs(t - tt))
        plt.plot(x, U_exp[n, :], "--", label=f"явная, t≈{t[n]:.2f}")
        plt.plot(x, U_imp[n, :], "-",  label=f"неявная, t≈{t[n]:.2f}")
    plt.xlabel("x")
    plt.ylabel("U")
    plt.title("Сравнение явной и неявной схем")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("slices_comparison.png", dpi=150)
    plt.close()


def plot_heatmap(x, t, U, name):
    plt.figure(figsize=(7, 4))
    plt.imshow(U, extent=[x[0], x[-1], t[0], t[-1]],
               origin="lower", aspect="auto")
    plt.colorbar(label="U")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(name.replace(" ", "_") + ".png", dpi=150)
    plt.close()



def main():
    print("=== Лабораторная работа №6 (вариант 26) ===")

    Nx = 80
    h = L / Nx
    Dmax = 11  # D(10)=11
    tau_max = h*h/(2*Dmax)
    Nt = int(T / tau_max) + 1

    print(f"Используем Nx={Nx}, Nt={Nt}, h={h:.4f}, tau≈{T/Nt:.3e}")

    # Численное решение
    x, t, U_exp = solve_explicit(Nx, Nt)
    _, _, U_imp = solve_implicit(Nx, Nt)

    plot_slices(x, t, U_exp, U_imp)
    plot_heatmap(x, t, U_imp, "Неявная схема — U(t,x)")

    convergence_experiment()


if __name__ == "__main__":
    main()
