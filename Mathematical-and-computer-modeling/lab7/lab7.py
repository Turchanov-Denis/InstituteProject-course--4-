import numpy as np
import matplotlib.pyplot as plt


# коэффициенты естественного роста популяций
r1, r2 = 2.0, 1.6

# ёмкости среды для каждого вида
K1, K2 = 200.0, 150.0

# порог, ниже которого считаем вид вымершим
eps_extinct = 1e-2


def rhs(N1, N2, alpha12, alpha21):
    """
    Правая часть системы дифференциальных уравнений.
    Возвращает скорости изменения численностей двух видов.
    """
    # уравнение роста первого вида с учётом конкуренции
    dN1 = r1 * N1 * (1 - (N1 + alpha12 * N2) / K1)

    # уравнение роста второго вида с учётом конкуренции
    dN2 = r2 * N2 * (1 - (N2 + alpha21 * N1) / K2)

    return dN1, dN2


def solve_rk4(alpha12, alpha21, N0=(50.0, 50.0), T=50.0, dt=0.02):
    """
    Численное решение системы методом Рунге–Кутты 4-го порядка.
    """
    # число шагов по времени
    steps = int(T / dt) + 1

    # массив временных значений
    t = np.linspace(0, T, steps)

    # массив для хранения численностей обоих видов
    N = np.zeros((steps, 2))

    # задание начальных условий
    N[0] = N0

    # основной цикл интегрирования
    for i in range(steps - 1):
        # текущие значения численностей
        N1, N2 = N[i]

        # коэффициенты метода Рунге–Кутты
        k1_1, k1_2 = rhs(N1, N2, alpha12, alpha21)
        k2_1, k2_2 = rhs(N1 + 0.5*dt*k1_1, N2 + 0.5*dt*k1_2, alpha12, alpha21)
        k3_1, k3_2 = rhs(N1 + 0.5*dt*k2_1, N2 + 0.5*dt*k2_2, alpha12, alpha21)
        k4_1, k4_2 = rhs(N1 + dt*k3_1,     N2 + dt*k3_2,     alpha12, alpha21)

        # вычисление следующего шага по формуле RK4
        N[i+1, 0] = N1 + (dt/6)*(k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
        N[i+1, 1] = N2 + (dt/6)*(k1_2 + 2*k2_2 + 2*k3_2 + k4_2)

        # защита от отрицательных значений из-за численных ошибок
        N[i+1] = np.maximum(N[i+1], 0.0)

    # возвращаем время и траектории численностей
    return t, N[:, 0], N[:, 1]


def equilibria(alpha12, alpha21):
    """
    Вычисление стационарных точек системы.
    """
    # список равновесий
    eq = []

    # тривиальные равновесия
    eq.append((0.0, 0.0))   # оба вида отсутствуют
    eq.append((K1, 0.0))    # существует только первый вид
    eq.append((0.0, K2))    # существует только второй вид

    # знаменатель формулы внутреннего равновесия
    denom = 1.0 - alpha12 * alpha21

    # проверка существования совместного равновесия
    if abs(denom) > 1e-12:
        N2_star = (K2 - alpha21 * K1) / denom
        N1_star = K1 - alpha12 * N2_star

        # проверка биологической осмысленности
        if N1_star > 0 and N2_star > 0:
            eq.append((N1_star, N2_star))

    return eq


def nullclines(alpha12, alpha21, n=200):
    """
    Построение изоклин системы.
    """
    # изоклина dN1/dt = 0
    N2_line = np.linspace(0, K2 * 1.2, n)
    N1_on_dN1 = K1 - alpha12 * N2_line

    # изоклина dN2/dt = 0
    N1_line = np.linspace(0, K1 * 1.2, n)
    N2_on_dN2 = K2 - alpha21 * N1_line

    return (N2_line, N1_on_dN1), (N1_line, N2_on_dN2)


def plot_time_and_phase(alpha12, alpha21, N0=(50.0, 50.0), T=50.0, dt=0.02):
    """
    Построение временных зависимостей и фазового портрета системы.
    """
    # численное решение системы
    t, N1, N2 = solve_rk4(alpha12, alpha21, N0, T, dt)

    # создание области для двух графиков
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # общий заголовок для рисунка
    fig.suptitle(
        f"Модель межвидовой конкуренции (α12 = {alpha12}, α21 = {alpha21})",
        fontsize=14
    )

    # ===== графики численностей во времени =====
    axs[0].plot(t, N1, label="Вид 1 (N1)")
    axs[0].plot(t, N2, "--", label="Вид 2 (N2)")
    axs[0].set_xlabel("Время t")
    axs[0].set_ylabel("Численность популяции")
    axs[0].set_title("Динамика численностей во времени")
    axs[0].legend()
    axs[0].grid()

    # ===== фазовый портрет =====
    ax = axs[1]

    # сетка для поля направлений
    N1g = np.linspace(0, K1 * 1.1, 21)
    N2g = np.linspace(0, K2 * 1.1, 21)
    X, Y = np.meshgrid(N1g, N2g)

    # вычисление поля направлений
    U, V = rhs(X, Y, alpha12, alpha21)
    speed = np.sqrt(U**2 + V**2)
    speed[speed == 0] = 1

    # поле направлений
    ax.quiver(X, Y, U/speed, V/speed)

    # фазовая траектория
    ax.plot(N1, N2, "k", linewidth=2, label="Фазовая траектория")

    # изоклины
    (N2l, N1l), (N1l2, N2l2) = nullclines(alpha12, alpha21)
    ax.plot(np.clip(N1l, 0, None), N2l, label="Изоклина dN1/dt = 0")
    ax.plot(N1l2, np.clip(N2l2, 0, None), label="Изоклина dN2/dt = 0")

    # стационарные точки
    for p in equilibria(alpha12, alpha21):
        ax.scatter(*p, s=50)

    ax.set_xlabel("Численность вида 1 (N1)")
    ax.set_ylabel("Численность вида 2 (N2)")
    ax.set_title("Фазовый портрет системы")
    ax.legend()
    ax.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # случай слабой конкуренции — устойчивое сосуществование
    plot_time_and_phase(alpha12=0.5, alpha21=0.5, N0=(60, 40), T=60)

    # вид 1 сильнее угнетает вид 2
    plot_time_and_phase(alpha12=0.5, alpha21=1.6, N0=(60, 40), T=60)

    # вид 2 сильнее угнетает вид 1
    plot_time_and_phase(alpha12=1.6, alpha21=0.5, N0=(60, 40), T=60)

    # сильная взаимная конкуренция — исход зависит от начальных условий
    plot_time_and_phase(alpha12=1.6, alpha21=1.6, N0=(60, 40), T=80)
