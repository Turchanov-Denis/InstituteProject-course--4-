"""
Лабораторная работа №5.
Вариант 28

Уравнение переноса:
    U_t + V(x) * U_x = f(x),   x ∈ [0, 10],  t ∈ [0, 100]

Дано:
    V(x) = x + 2
    f(x) ≡ -25.4  (правая часть — постоянная)
    U(0, x) = max(0, (x - 5)(10 - x))      — начальное условие
    U(t, 0) = t (t - 20)^2 / 200000        — левое граничное условие

1) аналитическое решение методом характеристик.
2) - явная схема 1-го порядка (явный «уголок», upwind),
   - неявная схема 1-го порядка (неявный «уголок», upwind).
3) среднеквадратичное отклонение (RMSE) между численным и
   аналитическим решениями на 5 разных сетках по пространству/времени.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import sqrt


out_dir = Path(__file__).parent
out_dir.mkdir(parents=True, exist_ok=True)

# Параметры задачи
L = 10.0
T = 100.0
f_const = -25.4


def V(x):
    """
    Функция скорости переноса V(x).
    Вариант 28: V(x) = x + 2.
    При V(x) > 0 характеристики идут вправо, поэтому граничное
    условие задаётся только слева (x = 0).
    """
    return x + 2.0


def phi_ic(x):
    """
    Начальное условие U(0, x).
        U(0, x) = max(0, (x - 5)(10 - x))
    """
    return np.maximum(0.0, (x - 5.0) * (10.0 - x))


def g_bc(t):
    """
    Граничное условие на левой границе x = 0.
        U(t, 0) = t (t - 20)^2 / 200000
    """
    return t * (t - 20.0) ** 2 / 200000.0


def analytic_U(t, x):
    """
    Аналитическое решение U(t,x) (метод характеристик).

    dx/dt = V(x) = x + 2  ->  (x + 2) e^{-t} = const

    Разделяющая кривая:
        t = ln((x+2)/2)

    - Если t <= ln((x+2)/2), характеристика идёт из начального слоя t=0.
    - Если t  > ln((x+2)/2), характеристика пришла с границы x=0.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    # Разделяющая кривая t = ln((x+2)/2)
    boundary = np.log((x + 2.0) / 2.0)

    from_ic = t <= boundary  # True -> используем начальные данные
    U = np.empty_like(t, dtype=float)

    # ---- 1) Область влияния начального условия ----
    if np.any(from_ic):
        # Точка на оси t=0, откуда пришла характеристика:
        # (x+2) e^{-t} = (x0+2) e^0 => x0 = (x+2)e^{-t} - 2
        xi0 = (x[from_ic] + 2.0) * np.exp(-t[from_ic]) - 2.0
        # Вдоль характеристики dU/dt = -25.4:
        # U(t,x) = -25.4 t + U(0,x0)
        U[from_ic] = f_const * t[from_ic] + phi_ic(xi0)

    # ---- 2) Область влияния граничного условия ----
    if np.any(~from_ic):
        # Время пересечения границы x=0:
        # (x+2)e^{-t} = 2 e^{-tb}  =>  tb = t - ln((x+2)/2)
        tb = t[~from_ic] - np.log((x[~from_ic] + 2.0) / 2.0)
        # По характеристике:
        # U(t,x) = U(tb,0) + f_const (t - tb)
        # t - tb = ln((x+2)/2)
        U[~from_ic] = g_bc(tb) + f_const * np.log((x[~from_ic] + 2.0) / 2.0)

    return U


def solve_explicit(Nx, Nt, L=L, T=T, f=f_const):
    """
    Явная upwind-схема 1-го порядка:

        (U_i^{n+1} - U_i^n)/tau + V_i (U_i^n - U_{i-1}^n)/h = f

        U_i^{n+1} = U_i^n - lambda_i (U_i^n - U_{i-1}^n) + tau f
        lambda_i = tau * V_i / h
    """
    h = L / Nx
    tau = T / Nt
    # Построение численной сетки
    x = np.linspace(0.0, L, Nx + 1)
    t = np.linspace(0.0, T, Nt + 1)

    Vx = V(x)

    U = np.zeros((Nt + 1, Nx + 1))

    # начальное условие
    U[0, :] = phi_ic(x)
    # левое граничное в момент t=0
    U[0, 0] = g_bc(0.0)

    for n in range(Nt): #шаги по времени
        # граничное условие на новом слое
        U[n + 1, 0] = g_bc(t[n + 1]) #Это условие на входе x = 0:

        # расчёт по x
        for i in range(1, Nx + 1):
            a = Vx[i]
            lam = tau * a / h

            U[n + 1, i] = U[n, i] - lam * (U[n, i] - U[n, i - 1]) + tau * f

    return x, t, U


def solve_implicit(Nx, Nt, L=L, T=T, f=f_const):
    """
    Неявная upwind-схема 1-го порядка:

        (U_i^{n+1} - U_i^n)/tau + V_i (U_i^{n+1} - U_{i-1}^{n+1})/h = f

        (1 + lambda_i) U_i^{n+1} - lambda_i U_{i-1}^{n+1} = U_i^n + tau f
        lambda_i = tau * V_i / h

    Система нижнетреугольная по i, решаем простым проходом.
    """
    h = L / Nx
    tau = T / Nt

    x = np.linspace(0.0, L, Nx + 1)
    t = np.linspace(0.0, T, Nt + 1)

    Vx = V(x)

    U = np.zeros((Nt + 1, Nx + 1))

    U[0, :] = phi_ic(x)
    U[0, 0] = g_bc(0.0)

    for n in range(Nt):
        U[n + 1, 0] = g_bc(t[n + 1])

        for i in range(1, Nx + 1):
            a = Vx[i]
            lam = tau * a / h

            U[n + 1, i] = (U[n, i] + tau * f + lam * U[n + 1, i - 1]) / (1.0 + lam)

    return x, t, U


def compute_rmse(U_num, U_exact):
    """
    RMSE = sqrt( (1/N) * Σ (U_num - U_exact)^2 )
    """
    return sqrt(np.mean((U_num - U_exact) ** 2))


def main():
    variants = [
        (10, 1110),
        (20, 2220),
        (40, 4440),
        (50, 5550),
        (100, 11100),
    ]

    print("Результаты сравнения (всё посчитано только кодом):")
    print(" Nx   Nt      RMSE_exp      RMSE_imp")
    print("--------------------------------------")

    table_lines = []

    # ----- Цикл по всем вариантам сетки -----
    # количество шагов по пространству , количество шагов по времени
    for Nx, Nt in variants:
        # Численное решение явной схемой
        x, t, U_exp = solve_explicit(Nx, Nt)
        # Численное решение неявной схемой
        _, _, U_imp = solve_implicit(Nx, Nt)

        # Аналитическое решение в узлах сетки
        TT, XX = np.meshgrid(t, x, indexing="ij")  # TT[n,i] = t^n, XX[n,i] = x_i
        U_an = analytic_U(TT, XX)

        # СКО для явной и неявной схем
        rmse_exp = compute_rmse(U_exp, U_an)
        rmse_imp = compute_rmse(U_imp, U_an)

        # Печать в консоль
        print(f"{Nx:3d} {Nt:5d}   {rmse_exp:10.5f}   {rmse_imp:10.5f}")
        # Для отчёта
        table_lines.append(f"{Nx:3d} {Nt:5d}   {rmse_exp:10.5f}   {rmse_imp:10.5f}")

    # ----- График при T=100 на достаточно мелкой сетке -----
    Nx, Nt = 100, 11100
    x, t, U_exp = solve_explicit(Nx, Nt)
    _, _, U_imp = solve_implicit(Nx, Nt)

    TT, XX = np.meshgrid(t, x, indexing="ij")
    U_an = analytic_U(TT, XX)

    U_exp_T = U_exp[-1, :]
    U_imp_T = U_imp[-1, :]
    U_an_T = U_an[-1, :]

    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot(x, U_an_T, label="аналитическое решение", linewidth=2)
    plt.plot(x, U_exp_T, "--", label="явная схема", linewidth=1.5)
    plt.plot(x, U_imp_T, ":", label="неявная схема", linewidth=1.5)
    plt.xlabel("x")
    plt.ylabel("U(T,x),  T = 100")
    plt.title("Сравнение аналитического и численных решений при T=100, Nx=100 (вариант 28)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    fig_path = out_dir / "compare_T100.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ----- Текстовый отчёт -----
    report_path = out_dir / "report_lab5_var28.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Сравнение среднеквадратических отклонений (RMSE)\n")
        f.write("Nx   Nt      RMSE_exp      RMSE_imp\n")
        f.write("--------------------------------------\n")
        for line in table_lines:
            f.write(line + "\n")



if __name__ == "__main__":
    main()
