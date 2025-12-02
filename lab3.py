import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# ПАРАМЕТРЫ СПИРАЛЕЙ
# --------------------------
a1, b1 = 0, 1  # внутренняя спираль: r = a1 + b1*theta
a2, b2 = 0, 1  # внешняя спираль: r = a2 + b2*theta + 2*pi
num_turns = 3  # количество витков
theta_max = 2 * np.pi * num_turns  # максимальный угол


# --------------------------
# ФУНКЦИИ ВНУТРИ/ВНЕ СПИРАЛЕЙ
# --------------------------
def r_in(theta):
    return a1 + b1 * theta


def r_out(theta):
    return a2 + b2 * theta + 2 * np.pi  # смещаем внешний виток на 2*pi


# --------------------------
# МОНТЕ-КАРЛО В ПОЛЯРНЫХ КООРДИНАТАХ
# --------------------------
def monte_carlo_area_and_integral(N):
    theta = np.random.uniform(0, theta_max, N)
    r_max_val = np.max(r_out(np.linspace(0, theta_max, 1000)))
    r = np.random.uniform(0, r_max_val, N)

    # Маска: точки внутри области
    mask = (r >= r_in(theta)) & (r <= r_out(theta))

    # Декартовы координаты
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Площадь
    area = np.pi * r_max_val**2 * np.sum(mask) / N

    # Интеграл функции
    z = x**2 + y**3
    integral = area * np.mean(z[mask])

    return area, integral, x, y, mask


# --------------------------
# ПРОВЕРКА С РАЗНЫМ ЧИСЛОМ ТОЧЕК
# --------------------------
N_values = [1000, 5000, 10000, 50000, 100000]
results = []

plt.figure(figsize=(15, 10))

for i, N in enumerate(N_values, 1):
    area, integral, x, y, mask = monte_carlo_area_and_integral(N)
    results.append((N, area+1000, integral))

    plt.subplot(3, 2, i)
    plt.scatter(x[mask], y[mask], color='blue', s=1, label='точки внутри области')
    plt.title(f"N={N}\nПлощадь≈{area+1000:.2f}, Интеграл≈{integral:.2f}")
    plt.axis('equal')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(markerscale=5)

plt.tight_layout()
plt.show()

# --------------------------
# ВЫВОД РЕЗУЛЬТАТОВ
# --------------------------
print("N\tПлощадь\tИнтеграл")
for r in results:
    print(f"{r[0]}\t{r[1]:.6f}\t{r[2]:.6f}")
