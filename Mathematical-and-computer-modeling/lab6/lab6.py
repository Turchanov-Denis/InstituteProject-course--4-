"""
Лабораторная работа №6. Модель диффузии. Вариант 26.

Численное решение уравнения:
    U_t = D(x) * U_xx - f(x) * U + 5
на отрезке 0 <= x <= 10, t in [0, T].

D(x) = x + 1
f(x) = x + 1

Начальное условие:
    U(0, x) = x^2 (10 - x)

Граничные условия:
    U(t, 0) = U(t, 10) = 0
- явная схема (вперёд по времени, центральная разность по x);
- неявная схема (назад по времени, центральная разность по x);
- исследование порядка аппроксимации (сравнение с эталонным решением);
"""

import numpy as np
import matplotlib.pyplot as plt
import os


L = 10.0   # длина отрезка по x
T = 1.0    # конечное время моделирования


#  Определение коэффициентов
def D(x: np.ndarray) -> np.ndarray:
    """
    Коэффициент диффузии D(x) = x + 1.

    Параметр:
        x : numpy.ndarray или float - координата(ы)

    Возвращает:
        numpy.ndarray или float - значение D(x)
    """
    return x + 1.0


def f_coef(x: np.ndarray) -> np.ndarray:
    """
    Коэффициент реакционного члена f(x) = x + 1.

    Параметр:
        x : numpy.ndarray или float

    Возвращает:
        f(x)
    """
    return x + 1.0


def source(x: np.ndarray, t: float) -> np.ndarray:
    """
    Источник в правой части уравнения. В задаче он постоянный: 5.

    Зависимость от x и t формально передана параметрами
    для удобства возможного изменения задачи.
    """
    return 5.0 * np.ones_like(x)


def initial_condition(x: np.ndarray) -> np.ndarray:
    """
    Начальное условие U(0, x) = x^2 (10 - x).
    """
    return x ** 2 * (10.0 - x)


def boundary_left(t: float) -> float:
    """Левая граница: U(t, 0) = 0."""
    return 0.0


def boundary_right(t: float) -> float:
    """Правая граница: U(t, 10) = 0."""
    return 0.0


def solve_explicit(N: int, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Решение уравнения явной схемой.

    Пространственная сетка: N отрезков, N+1 узел.
    Шаг по времени tau (количество шагов подбирается так, чтобы
    дойти примерно до времени T).

    условием устойчивости:
        tau <= h^2 / (2 * max(D(x)))
    Здесь max(D) = 11, так как x in [0, 10], D(x) = x + 1.

    Параметры:
        N   - число отрезков по x (узлов N+1)
        tau - шаг по времени

    Возвращает:
        x   - массив узлов по x (N+1)
        t   - массив моментов времени (M+1)
        U   - матрица решений размера (M+1) x (N+1)
              U[n, i] ~ U(t^n, x_i)
    """
    h = L / N  # шаг по x

    # Оценка числа шагов по времени, чтобы дойти до T
    M = int(np.round(T / tau))
    tau = T / M  # слегка корректируем tau, чтобы ровно попасть в T

    # Сетка
    x = np.linspace(0.0, L, N + 1)
    t = np.linspace(0.0, T, M + 1)

    # Значения коэффициентов на узлах по x
    D_vals = D(x)
    f_vals = f_coef(x)

    # Матрица решения
    U = np.zeros((M + 1, N + 1))

    # Начальное условие
    U[0, :] = initial_condition(x)

    # Явная формула:
    # U_i^{n+1} = U_i^n + tau * ( D_i * (U_{i+1}^n - 2U_i^n + U_{i-1}^n) / h^2
    #                             - f_i * U_i^n + 5 )
    for n in range(0, M):
        # Граничные условия (они равны нулю, но записываем явно)
        U[n, 0] = boundary_left(t[n])
        U[n, -1] = boundary_right(t[n])

        # Внутренние узлы: i = 1..N-1
        U_n = U[n, :]
        laplace = (U_n[2:] - 2.0 * U_n[1:-1] + U_n[:-2]) / h**2
        reaction = -f_vals[1:-1] * U_n[1:-1]
        rhs = D_vals[1:-1] * laplace + reaction + 5.0

        U[n + 1, 1:-1] = U_n[1:-1] + tau * rhs

        # Границы в новом слое
        U[n + 1, 0] = boundary_left(t[n + 1])
        U[n + 1, -1] = boundary_right(t[n + 1])

    return x, t, U


#  Неявная схема (трёхдиагональная СЛАУ)
def thomas_algorithm(a: np.ndarray, b: np.ndarray,
                     c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Метод прогонки (Алгоритм Томаса) для решения трёхдиагональной СЛАУ.

    Система имеет вид:
        a_i * y_{i-1} + b_i * y_i + c_i * y_{i+1} = d_i,  i = 0..n-1
    Предполагается a_0 = 0, c_{n-1} = 0.

    Параметры:
        a, b, c - под-, главная и над-диагонали (длины n)
        d       - правая часть (длины n)

    Возвращает:
        y       - решение (длины n)
    """
    n = len(d)
    # Копии, чтобы не портить исходные массивы
    ac, bc, cc, dc = map(np.array, (a, b, c, d))

    # Прямой ход: модификация коэффициентов
    for i in range(1, n):
        mc = ac[i] / bc[i - 1]
        bc[i] = bc[i] - mc * cc[i - 1]
        dc[i] = dc[i] - mc * dc[i - 1]

    # Обратный ход
    y = np.zeros(n)
    y[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        y[i] = (dc[i] - cc[i] * y[i + 1]) / bc[i]

    return y


def solve_implicit(N: int, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Решение уравнения неявной схемой (назад по времени).

    Параметры:
        N   - число отрезков по x (узлов N+1)
        tau - шаг по времени (разрешается брать любой, устойчивость обеспечена)

    Возвращает:
        x, t, U - аналогично функции solve_explicit
    """
    h = L / N
    M = int(np.round(T / tau))
    tau = T / M  # корректируем tau
    x = np.linspace(0.0, L, N + 1)
    t = np.linspace(0.0, T, M + 1)

    D_vals = D(x)
    f_vals = f_coef(x)

    U = np.zeros((M + 1, N + 1))
    U[0, :] = initial_condition(x)

    # Число внутренних узлов
    K = N - 1

    # Подготовим диагонали матрицы A для внутренних узлов.
    # Для узла i (по x), i = 1..N-1 (всего K узлов):
    # -tau*D_i/h^2 * U_{i-1}^{n+1}
    # + (1 + 2*tau*D_i/h^2 + tau*f_i) * U_i^{n+1}
    # -tau*D_i/h^2 * U_{i+1}^{n+1}
    # = U_i^n + tau*5
    #
    # При этом U_0^{n+1} и U_N^{n+1} = 0, поэтому дополнительных слагаемых в правой части нет.
    #
    # Составим диагонали для индексов j = 0..K-1, где j соответствует узлу i=j+1.

    x_inner = x[1:-1]
    D_inner = D_vals[1:-1]
    f_inner = f_vals[1:-1]

    alpha = -tau * D_inner / h**2           # под- и над-диагональ
    beta = 1.0 + 2.0 * tau * D_inner / h**2 + tau * f_inner  # главная диагональ

    # Под-диагональ a, над-диагональ c и главная диагональ b
    a = np.zeros(K)
    b = np.zeros(K)
    c = np.zeros(K)

    b[:] = beta
    a[1:] = alpha[1:]    # a[0] = 0
    c[:-1] = alpha[:-1]  # c[K-1] = 0

    for n in range(0, M):
        # Граничные значения на текущем слое (для полноты, хотя они нули)
        U[n, 0] = boundary_left(t[n])
        U[n, -1] = boundary_right(t[n])

        # Правая часть
        d = U[n, 1:-1] + tau * source(x_inner, t[n + 1])

        # Решаем трёхдиагональную систему
        U_inner_next = thomas_algorithm(a, b, c, d)

        # Записываем результат
        U[n + 1, 1:-1] = U_inner_next
        U[n + 1, 0] = boundary_left(t[n + 1])
        U[n + 1, -1] = boundary_right(t[n + 1])

    return x, t, U


#  Исследование порядка аппроксимации

def compute_error_against_reference(method: str,
                                    N_list: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Оценивает погрешность схемы (явной или неявной) по отношению
    к "эталонному" решению на очень мелкой сетке.

    Параметры:
        method : "explicit" или "implicit" - какую схему исследуем
        N_list : список размеров сетки по x (например [20, 40, 80, 160])

    Возвращает:
        h_vals     - массив шагов по x
        error_vals - соответствующие ошибки (норма max по x в момент T)
    """
    # Сначала строим эталонное решение (неявной схемой на очень мелкой сетке)
    N_ref = 640  # достаточно мелкая сетка
    h_ref = L / N_ref

    # Для уменьшения временной погрешности возьмём tau_ref ~ h_ref^2
    D_max = 11.0
    sigma = 0.25  # запас по устойчивости
    tau_ref = sigma * h_ref**2 / D_max

    x_ref, t_ref, U_ref = solve_implicit(N_ref, tau_ref)
    U_ref_final = U_ref[-1, :]

    h_vals = []
    error_vals = []

    for N in N_list:
        h = L / N
        h_vals.append(h)

        # Чтобы узлы грубой сетки были подмножеством узлов эталонной,
        # N_ref должен делиться на N. Мы так подобрали значения.
        factor = N_ref // N

        # Выбор tau в зависимости от метода
        if method == "explicit":
            # Для явной схемы tau ограничен сверху условием устойчивости:
            # tau <= h^2 / (2 * D_max). Возьмём с запасом,
            # а также поставим tau ~ h^2, чтобы пространственная и временная
            # погрешности были одинакового порядка.
            sigma = 0.4
            tau = sigma * h**2 / D_max
            x, t, U = solve_explicit(N, tau)
        elif method == "implicit":
            # Для неявной схемы устойчивость безусловная.
            # Также возьмём tau ~ h^2 для баланса ошибок.
            sigma = 0.4
            tau = sigma * h**2 / D_max
            x, t, U = solve_implicit(N, tau)
        else:
            raise ValueError("method должен быть 'explicit' или 'implicit'")

        U_final = U[-1, :]

        # Сравниваем значения только в узлах грубой сетки.
        # Индексы узлов грубой сетки на эталонной: каждые 'factor' узлов.
        U_ref_on_coarse = U_ref_final[::factor]

        # Вычисляем норму max по x
        err = np.max(np.abs(U_final - U_ref_on_coarse))
        error_vals.append(err)

    return np.array(h_vals), np.array(error_vals)


#  Построение и сохранение графиков
def plot_solutions(x, t, U, title_prefix: str, filename_prefix: str) -> None:
    """
    Строит несколько срезов решения по времени и сохраняет график.

    Параметры:
        x, t, U         - сетка и решение
        title_prefix    - текст в заголовке
        filename_prefix - префикс имени файла PNG
    """
    # Выберем несколько слоёв по времени
    time_indices = [0, len(t)//4, len(t)//2, len(t)-1]

    plt.figure(figsize=(8, 5))
    for idx in time_indices:
        plt.plot(x, U[idx, :], label=f"t = {t[idx]:.3f}")
    plt.xlabel("x")
    plt.ylabel("U(x, t)")
    plt.title(f"{title_prefix}: несколько срезов по времени")
    plt.legend()
    plt.grid(True)

    filename = f"{filename_prefix}_time_slices.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"График сохранён в файл: {filename}")


def plot_compare_final(x, U1_final, U2_final,
                       label1: str, label2: str,
                       filename: str) -> None:
    """
    Сравнение двух решений в конечный момент времени.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(x, U1_final, marker="o", linestyle="-", label=label1)
    plt.plot(x, U2_final, marker="x", linestyle="--", label=label2)
    plt.xlabel("x")
    plt.ylabel("U(x, T)")
    plt.title("Сравнение решений в момент времени T")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"График сохранён в файл: {filename}")


def plot_error(h_vals, err_vals, method: str, filename: str) -> None:
    """
    Логарифмический график ошибки от шага h. По наклону зависимости
    log(err) ~ p * log(h) можно оценить порядок p.
    """
    plt.figure(figsize=(6, 5))
    plt.loglog(h_vals, err_vals, marker="o")
    plt.xlabel("h (шаг по x)")
    plt.ylabel("ошибка, ||U_h - U_ref||_inf")
    plt.title(f"Порядок аппроксимации, метод: {method}")
    plt.grid(True, which="both")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"График сохранён в файл: {filename}")

    # Оценка наклона по двум последним точкам
    if len(h_vals) >= 2:
        p = (np.log(err_vals[-1]) - np.log(err_vals[-2])) / \
            (np.log(h_vals[-1]) - np.log(h_vals[-2]))
        print(f"Оценка порядка по двум последним точкам для метода {method}: p ≈ {p:.2f}")


def main():
    # Создадим папку (опционально) для картинок
    # Если не нужно отдельной папки, можно закомментировать.
    img_dir = os.getcwd()  # текущая папка
    print(f"Графики будут сохранены в каталог: {img_dir}")

    #  Одно решение явной схемой
    N = 100
    h = L / N
    D_max = 11.0

    # Выбор tau согласно условию устойчивости явной схемы:
    sigma = 0.4  # запас (должно быть <= 0.5)
    tau_explicit = sigma * h**2 / D_max
    print(f"Явная схема: N={N}, h={h:.4f}, tau={tau_explicit:.6e}")

    x_exp, t_exp, U_exp = solve_explicit(N, tau_explicit)
    plot_solutions(x_exp, t_exp, U_exp,
                   title_prefix="Явная схема",
                   filename_prefix="explicit")

    #  Одно решение неявной схемой
    tau_implicit = sigma * h**2 / D_max  # берём такой же закон tau ~ h^2
    print(f"Неявная схема: N={N}, h={h:.4f}, tau={tau_implicit:.6e}")

    x_imp, t_imp, U_imp = solve_implicit(N, tau_implicit)
    plot_solutions(x_imp, t_imp, U_imp,
                   title_prefix="Неявная схема",
                   filename_prefix="implicit")

    #  Сравнение явной и неявной схем в момент T
    # Для корректного сравнения нужно удостовериться, что сетки по x совпадают
    # (мы использовали одинаковое N, поэтому x_exp == x_imp).
    filename_compare = "compare_explicit_implicit_T.png"
    plot_compare_final(x_exp, U_exp[-1, :], U_imp[-1, :],
                       label1="явная схема",
                       label2="неявная схема",
                       filename=filename_compare)

    #  Исследование порядка аппроксимации
    N_list = [20, 40, 80, 160]

    # Явная схема
    h_vals_exp, err_vals_exp = compute_error_against_reference(
        method="explicit",
        N_list=N_list
    )
    plot_error(h_vals_exp, err_vals_exp,
               method="явная схема",
               filename="error_order_explicit.png")

    # Неявная схема
    h_vals_imp, err_vals_imp = compute_error_against_reference(
        method="implicit",
        N_list=N_list
    )
    plot_error(h_vals_imp, err_vals_imp,
               method="неявная схема",
               filename="error_order_implicit.png")


if __name__ == "__main__":
    main()
