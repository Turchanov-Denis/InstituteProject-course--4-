"""
Лабораторная работа №4. Вариант 13
Тема: Стохастические модели — случайные блуждания и IFS
Автор: <впиши своё ФИО>

Цели работы:
1. Смоделировать случайные блуждания на квадратной решётке с «прилипанием»
   к заданной точке и уже прилипшим точкам.
2. Построить IFS-аттрактор по трём заданным аффинным отображениям.
3. Для полученных множеств оценить метрическую (фрактальную) размерность
   методом боксового счёта (box counting).
4. Визуализировать результаты моделирования.
"""

# ============================================================
# 1. Импорт необходимых библиотек
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, log
import random

# Для воспроизводимости результатов фиксируем генераторы случайных чисел
random.seed(13)
np.random.seed(13)

# ============================================================
# 2. Функции для метода боксового счёта (Box Counting)
# ============================================================

def box_counting_dimension(points, eps_list=None):
    """
    Оценка фрактальной (метрической) размерности множества точек
    методом боксового счёта.

    Идея метода:
    ---------------------------------------------
    1. Пространство разбивается на сетку квадратов (боксов) со стороной ε.
    2. Считается количество занятых боксов N(ε), т.е. тех,
       в которых есть хотя бы одна точка.
    3. Меняем ε, строим зависимость log N(ε) от log(1/ε).
    4. При фрактальном поведении точки примерно лежат на прямой:
           log N(ε) ≈ D * log(1/ε) + const
       где D — искомая фрактальная размерность.

    Параметры:
        points : array_like (n,2)
        eps_list : список масштабов ε

    Возвращает:
        D — оценка размерности
        eps_list, logs_eps, logs_N — данные для построения графика
    """

    pts = np.asarray(points)
    if pts.size == 0:
        return np.nan, [], [], []

    # Сдвигаем точки к (0,0) — это не влияет на размерность,
    # но упрощает работу с индексами боксов
    mn = pts.min(axis=0)
    pts0 = pts - mn

    # Если eps_list не задан — строим автоматически на основе размера множества
    if eps_list is None:
        span = (pts0.max(axis=0) - pts0.min(axis=0)).max()
        max_pow = int(np.floor(np.log2(max(span, 1))))
        eps_list = [2 ** k for k in range(max_pow, 0, -1)]

    logs_eps = []  # log(1/ε)
    logs_N = []    # log N(ε)

    for eps in eps_list:
        # Разбиваем пространство на квадраты ε×ε
        idx = np.floor(pts0 / eps).astype(int)

        # Кол-во уникальных ячеек — это N(ε)
        n_boxes = len(np.unique(idx, axis=0))
        if n_boxes > 0:
            logs_eps.append(log(1.0 / eps))
            logs_N.append(log(n_boxes))

    # Если данных мало — оценка невозможна
    if len(logs_eps) < 2:
        return np.nan, eps_list, logs_eps, logs_N

    # Оцениваем наклон прямой log N = D * log(1/ε) + const (метод МНК)
    A = np.vstack([logs_eps, np.ones(len(logs_eps))]).T
    D, _ = np.linalg.lstsq(A, logs_N, rcond=None)[0]
    return D, eps_list, logs_eps, logs_N


def plot_boxcount(logs_eps, logs_N, title="Box counting"):
    """
    Строит график зависимости log N(ε) от log(1/ε).
    По наклону прямой оценивается фрактальная размерность (метод box-counting).
    """
    plt.figure(figsize=(6, 4), dpi=130)
    plt.plot(logs_eps, logs_N, "o-", lw=1)
    plt.xlabel("log(1/ε)")
    plt.ylabel("log N(ε)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    # --- сохраняем в файл в папку запуска ---
    plt.savefig(title.replace(" ", "_") + ".png", dpi=200, bbox_inches="tight")

    plt.show()


# ============================================================
# 3. Задание I — генерация случайного кластера (DLA-тип)
# ============================================================

def random_walk_cluster(n_particles=2000, grid_half=150, stick_point=(0, 0)):
    """
    Генерирует кластер из точек, образованный случайными блужданиями
    и прилипанием к уже занятым узлам.
    """

    occ = {stick_point}  # начальный кластер — одна точка
    radius = 1           # максимальный достигнутый радиус кластера

    def spawn(r):
        """Старт новой частицы на окружности чуть дальше текущего кластера."""
        ang = random.random() * 2 * np.pi
        R = max(5, r + 5)
        x = int(R * np.cos(ang))
        y = int(R * np.sin(ang))
        return [x, y]

    def near_cluster(x, y):
        """Проверка 4-соседей — критерий прилипания."""
        return ((x + 1, y) in occ or (x - 1, y) in occ or
                (x, y + 1) in occ or (x, y - 1) in occ)

    for _ in range(n_particles):
        x, y = spawn(radius)  # старт частицы

        while True:
            # Случайный шаг
            d = random.randint(0, 3)
            if d == 0: x += 1
            elif d == 1: x -= 1
            elif d == 2: y += 1
            else: y -= 1

            # Если рядом с кластером — прилипает
            if near_cluster(x, y):
                occ.add((x, y))
                r2 = x * x + y * y
                if r2 > radius * radius:
                    radius = int(sqrt(r2)) + 1
                break

            # Если ушла слишком далеко — запускаем заново
            if abs(x) > grid_half or abs(y) > grid_half:
                x, y = spawn(radius)

    return occ


def plot_cluster(points, title="Кластер DLA"):
    """Строит изображение построенного кластера."""
    pts = np.array(list(points))
    plt.figure(figsize=(6, 6), dpi=130)
    plt.scatter(pts[:, 0], pts[:, 1], s=1)
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()

    # --- сохраняем ---
    plt.savefig(title.replace(" ", "_") + ".png", dpi=200, bbox_inches="tight")

    plt.show()


# ============================================================
# 4. Задание II — IFS и хаос-игра
# ============================================================

# Матрицы и векторы из условия варианта 13
A1 = np.array([[-0.169, -0.848],
               [-0.296,  0.192]], float)
b1 = np.array([26.635, 67.988])

A2 = np.array([[ 0.320, -0.788],
               [ 0.548,  0.288]], float)
b2 = np.array([22.636, -7.129])

A3 = np.array([[ 0.417, -0.392],
               [-0.740,  0.208]], float)
b3 = np.array([-49.880, -63.200])

As = [A1, A2, A3]
bs = [b1, b2, b3]


def chaos_game(A_list, b_list, p=None, n=100_000, burn=1000, clip_R=1e6):
    """
    Строит IFS-аттрактор хаос-игрой:
        x_{k+1} = A_i x_k + b_i
    Индекс i выбирается случайно.
    """
    if p is None:
        p = np.ones(len(A_list)) / len(A_list)

    cdf = np.cumsum(p)
    x = np.zeros(2)
    pts = []

    # burn-in — пропускаем переходный процесс
    for _ in range(burn):
        r = random.random()
        i = int(np.searchsorted(cdf, r))
        x = A_list[i] @ x + b_list[i]
        if np.linalg.norm(x) > clip_R:
            x[:] = 0.0

    # сохраняем точки
    for _ in range(n):
        r = random.random()
        i = int(np.searchsorted(cdf, r))
        x = A_list[i] @ x + b_list[i]
        if np.linalg.norm(x) > clip_R:
            x[:] = 0.0
        pts.append(x.copy())

    return np.array(pts)


def plot_points(pts, title="IFS-аттрактор"):
    """Отображает точки IFS-аттрактора."""
    plt.figure(figsize=(7, 6), dpi=130)
    plt.scatter(pts[:, 0], pts[:, 1], s=0.2)
    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()

    # --- сохраняем ---
    plt.savefig(title.replace(" ", "_") + ".png", dpi=200, bbox_inches="tight")

    plt.show()


# ============================================================
# 5. Основной блок программы — моделирование
# ============================================================

if __name__ == "__main__":

    print("=== ЗАДАНИЕ I — случайные блуждания ===")

    particle_counts = [1000, 3000, 6000]

    for n in particle_counts:
        print(f"\nМоделирование DLA-кластера для n = {n}")
        occ = random_walk_cluster(n_particles=n, grid_half=150, stick_point=(0, 0))
        pts = np.array(list(occ))

        # Box-counting оценка размерности
        D, eps_list, logs_eps, logs_N = box_counting_dimension(
            pts,
            eps_list=[1, 2, 4, 8, 16, 32, 64]
        )
        print(f"  Оценка фрактальной размерности: D ≈ {D:.4f}")

        # Визуализация кластера
        plot_cluster(pts, title=f"DLA_{n}")

        # Визуализация графика box-counting
        plot_boxcount(logs_eps, logs_N, title=f"DLA_box_{n}")

    print("\n=== ЗАДАНИЕ II — IFS ===")

    point_counts = [50_000, 150_000, 300_000]

    for n in point_counts:
        print(f"\nГенерация IFS-аттрактора для N = {n}")
        pts2 = chaos_game(As, bs, n=n, burn=2000)

        D2, eps_list2, logs_eps2, logs_N2 = box_counting_dimension(pts2)
        print(f"  Оценка фрактальной размерности IFS: D ≈ {D2:.4f}")

        plot_points(pts2, title=f"IFS_{n}")
        plot_boxcount(logs_eps2, logs_N2, title=f"IFS_box_{n}")

    print("\n--- моделирование завершено ---")
