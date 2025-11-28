import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def two_body_elastic(u1, u2, m1, m2): #
    """Возвращает скорости после полного упругого соударения двух тел (по одной оси)."""
    v1 = ((m1 - m2) / (m1 + m2)) * u1 + (2 * m2 / (m1 + m2)) * u2
    v2 = (2 * m1 / (m1 + m2)) * u1 + ((m2 - m1) / (m1 + m2)) * u2
    return v1, v2

def cascade_final_velocities(N, q, Vb=1.0, m1=1.0):
    """
    Моделирует каскад из N шаров: m_i = m1 * q^(i-1), начальные скорости: v0 = Vb, остальные 0.
    Возвращает список окончательных скоростей (v1..vN).
    """
    masses = np.array([m1 * q**(i) for i in range(N)])  # i от 0 до N-1
    velocities = np.zeros(N)
    velocities[0] = Vb
    for i in range(N-1):
        u1 = velocities[i]
        u2 = velocities[i+1]
        m1_i = masses[i]
        m2_i = masses[i+1]
        v1, v2 = two_body_elastic(u1, u2, m1_i, m2_i)
        velocities[i] = v1
        velocities[i+1] = v2
    return masses, velocities

N = 9
Vb = 1.0
m1 = 1.0

# построим график Ve/Vb в зависимости от q
qs = np.linspace(0.01, 1.0, 400)
Ve_over_Vb = []
for q in qs:
    masses, velocities = cascade_final_velocities(N, q, Vb=Vb, m1=m1)
    Ve_over_Vb.append(velocities[-1] / Vb)

Ve_over_Vb = np.array(Ve_over_Vb)

plt.figure(figsize=(8,4.5))
plt.plot(qs, Ve_over_Vb)
plt.xlabel('q (m_{i+1} = q * m_i)')
plt.ylabel('V_e / V_b')
plt.title(f'Отношение V_e/V_b для N={N} шаров (каскадный режим)')
plt.grid(True)
plt.ylim(0, max(1.1*Ve_over_Vb.max(), 1.0))
plt.show()



table_qs = np.array([0.5])
rows = []
for q in table_qs:
    masses, velocities = cascade_final_velocities(N, q, Vb=Vb, m1=m1)
    rows.append({'q': q, 'V_e/V_b': velocities[-1]/Vb, **{f'v_{i+1}': velocities[i] for i in range(N)}})

df = pd.DataFrame(rows)
df = df[['q', 'V_e/V_b'] + [f'v_{i+1}' for i in range(N)]]

# также вывести DataFrame в вывод
df = pd.DataFrame(rows)
cols_order = ['q', 'V_e/V_b'] + [f'v_{i+1}' for i in range(N)]
df = df[cols_order]
print("\nТаблица результатов (примерные значения):")
print(df.to_string(index=False))

# csv_fname = os.path.join("cascade_results_table.csv")
# df.to_csv(csv_fname, index=False)

