import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# ВХОДНЫЕ ДАННЫЕ (Вариант 10)
# -----------------------------
lam = 0.029         # длина волны λ = 2.9 см
d = 0.02            # расстояние между щелями = 2 см
N = 14              # количество щелей (тип 1 по табл. 4)

k = 2 * np.pi / lam  # волновое число

# Угловая сетка: DIAGRAM FOR θ ∈ [–90°, +90°]
theta_deg = np.linspace(-90, 90, 2000)
theta = np.deg2rad(theta_deg)

# ----------------------------------
# ЭЛЕМЕНТНА ДІАГРАМА (приближение)
# ----------------------------------
# Для продольных щелей в широкой стенке:
# F_el(θ) ≈ |cosθ|
def F_element(theta):
    return np.abs(np.cos(theta))

# ----------------------------------
# МНОЖИТЕЛЬ РЕШЕТКИ
# ----------------------------------
# AF(θ) = sin(N*ψ/2) / (N*sin(ψ/2)),
# ψ = k*d*sin(θ)
def array_factor(theta):
    psi = k * d * np.sin(theta)
    psi_half = psi / 2

    num = np.sin(N * psi_half)
    den = N * np.sin(psi_half)

    den = np.where(np.abs(den) < 1e-9, 1e-9, den)

    AF = np.abs(num / den)
    return AF

# ----------------------------------
# ПОЛНАЯ ДИАГРАММА В H И E
# ----------------------------------
F_H = F_element(theta) * array_factor(theta)
F_H_norm = F_H / np.max(F_H)

# Для E-плоскости вводим дополнительный cosθ
F_E = F_element(theta) * np.abs(np.cos(theta)) * array_factor(theta)
F_E_norm = F_E / np.max(F_E)

# ----------------------------------
# HPBW 0.707
# ----------------------------------
def HPBW(pattern, theta):
    patt = pattern / np.max(pattern)
    target = 1 / np.sqrt(2)

    i0 = np.argmax(patt)

    left = None
    for i in range(i0, 0, -1):
        if patt[i] >= target and patt[i - 1] < target:
            left = theta[i]
            break

    right = None
    for i in range(i0, len(patt) - 1):
        if patt[i] >= target and patt[i + 1] < target:
            right = theta[i]
            break

    if left is None or right is None:
        return None

    return np.degrees(right - left)

# ----------------------------------
# Уровень первого бокового лепестка
# ----------------------------------
def first_sidelobe_level(pattern, theta):
    patt = pattern / np.max(pattern)
    i0 = np.argmax(patt)

    start = i0 + 20
    if start >= len(patt):
        return None, None

    sub = patt[start:]
    i_local = np.argmax(sub)
    idx = start + i_local

    angle = np.degrees(theta[idx])
    level = patt[idx]
    level_db = 20 * np.log10(level)

    return angle, level_db

# ----------------------------------
# Расчет параметров
# ----------------------------------
hpbw_H = HPBW(F_H_norm, theta)
hpbw_E = HPBW(F_E_norm, theta)

theta_sl_H, sl_H_db = first_sidelobe_level(F_H_norm, theta)
theta_sl_E, sl_E_db = first_sidelobe_level(F_E_norm, theta)

print(f"HPBW (H-плоскость): {hpbw_H:.2f} град")
print(f"HPBW (E-плоскость): {hpbw_E:.2f} град")
print(f"Первый боковой лепесток H: угол ≈ {theta_sl_H:.1f}°, уровень ≈ {sl_H_db:.1f} дБ")
print(f"Первый боковой лепесток E: угол ≈ {theta_sl_E:.1f}°, уровень ≈ {sl_E_db:.1f} дБ")

# ----------------------------------
# Графики
# ----------------------------------
plt.figure(figsize=(10, 6))
plt.plot(theta_deg, F_H_norm, label="H-площина")
plt.axhline(0.707, linestyle="--", color="red")
plt.xlabel("θ, град")
plt.ylabel("Нормована ДС")
plt.title("ДС ХвЩА у H-площині (варіант 10)")
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 6))
plt.plot(theta_deg, F_E_norm, label="E-площина")
plt.axhline(0.707, linestyle="--", color="red")
plt.xlabel("θ, град")
plt.ylabel("Нормована ДС")
plt.title("ДС ХвЩА у E-площині (варіант 10)")
plt.grid(True)
plt.legend()

plt.show()
