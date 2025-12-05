import random
import matplotlib.pyplot as plt
import math

# ==========================
# ЗАДАНИЕ I: Случайное блуждание с прилипанием
# ==========================

def random_walk_sticky(grid_size=100, steps=5000):
    """
    Моделируем случайное блуждание с прилипанием на квадратной сетке.
    grid_size: размер сетки (grid_size x grid_size)
    steps: количество шагов случайного блуждания
    """

    # Создаём пустую сетку (0 = не прилипло, 1 = прилипло)
    grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

    # Начальная точка - центр сетки
    x, y = grid_size // 2, grid_size // 2
    grid[y][x] = 1  # точка сразу прилипает
    sticky_points = {(x, y)}  # множество прилипших точек

    for _ in range(steps):
        # Выбираем случайное направление: вверх, вниз, влево, вправо
        dx, dy = random.choice([(0,1),(0,-1),(1,0),(-1,0)])
        x_new, y_new = x + dx, y + dy

        # Проверяем, что не вышли за границы
        x_new = max(0, min(grid_size-1, x_new))
        y_new = max(0, min(grid_size-1, y_new))

        # Проверяем, есть ли соседняя прилипшая точка
        neighbors = [(x_new+dx2, y_new+dy2) for dx2, dy2 in [(0,1),(0,-1),(1,0),(-1,0)]]
        stuck = any((nx, ny) in sticky_points for nx, ny in neighbors if 0 <= nx < grid_size and 0 <= ny < grid_size)

        if stuck:
            grid[y_new][x_new] = 1
            sticky_points.add((x_new, y_new))

        x, y = x_new, y_new

    # Визуализация
    xs, ys = zip(*sticky_points)
    plt.figure(figsize=(6,6))
    plt.scatter(xs, ys, s=1, c='black')
    plt.title(f"Случайное блуждание с прилипанием, шагов={steps}")
    plt.show()

    # ==========================
    # Вычисление метрической размерности методом коробок
    # ==========================
    def box_counting(points, box_sizes):
        counts = []
        for box_size in box_sizes:
            boxes = set()
            for px, py in points:
                box_x = px // box_size
                box_y = py // box_size
                boxes.add((box_x, box_y))
            counts.append(len(boxes))
        return counts

    box_sizes = [1, 2, 4, 8, 16, 32]  # размеры коробок
    counts = box_counting(sticky_points, box_sizes)
    log_counts = [math.log(c) for c in counts]
    log_sizes = [math.log(1/bs) for bs in box_sizes]

    # Оценка наклона прямой (приближение D)
    D = (log_counts[-1] - log_counts[0]) / (log_sizes[-1] - log_sizes[0])
    print(f"Примерная метрическая размерность блуждания: {D:.3f}")

# ==========================
# ЗАДАНИЕ II: Аттрактор из матриц
# ==========================

def linear_attractor(matrices, vectors, points_num=10000):
    """
    Построение аттрактора из линейных матриц и векторов.
    matrices: список 2x2 матриц
    vectors: список векторов длины 2
    points_num: количество итераций
    """

    # Начальная точка
    x, y = 0.0, 0.0
    points = []

    for _ in range(points_num):
        # Случайно выбираем одно из отображений
        i = random.randint(0, len(matrices)-1)
        A = matrices[i]
        b = vectors[i]

        # Линейное отображение: x_{n+1} = A * x_n + b
        x_new = A[0][0]*x + A[0][1]*y + b[0]
        y_new = A[1][0]*x + A[1][1]*y + b[1]

        points.append((x_new, y_new))
        x, y = x_new, y_new

    # Визуализация
    xs, ys = zip(*points)
    plt.figure(figsize=(6,6))
    plt.scatter(xs, ys, s=0.5, c='blue')
    plt.title(f"Аттрактор с {points_num} точками")
    plt.show()

    # ==========================
    # Метрическая размерность методом коробок
    # ==========================
    def box_counting(points, box_sizes):
        counts = []
        # Сдвигаем точки, чтобы все были положительными
        min_x = min(px for px, py in points)
        min_y = min(py for px, py in points)
        shifted_points = [(px - min_x, py - min_y) for px, py in points]

        for box_size in box_sizes:
            boxes = set()
            for px, py in shifted_points:
                box_x = int(px // box_size)
                box_y = int(py // box_size)
                boxes.add((box_x, box_y))
            counts.append(len(boxes))
        return counts

    # выбираем размер коробок исходя из диапазона точек
    max_coord = max(max(abs(px), abs(py)) for px, py in points)
    box_sizes = [max_coord/(2**i) for i in range(1, 7)]
    counts = box_counting(points, box_sizes)
    log_counts = [math.log(c) for c in counts]
    log_sizes = [math.log(1/bs) for bs in box_sizes]

    D = (log_counts[-1] - log_counts[0]) / (log_sizes[-1] - log_sizes[0])
    print(f"Примерная метрическая размерность аттрактора: {D:.3f}")

# ==========================
# Запуск примеров
# ==========================

# Задание I
random_walk_sticky(grid_size=100, steps=5000)

# Задание II
matrices = [
    [[-0.169, -0.848], [-0.296, 0.192]],
    [[0.320, -0.788], [0.548, 0.288]],
    [[0.417, -0.392], [-0.740, 0.208]]
]
vectors = [
    [26.635, 67.988],
    [22.636, -7.129],
    [-49.880, -63.200]
]
linear_attractor(matrices, vectors, points_num=10000)
