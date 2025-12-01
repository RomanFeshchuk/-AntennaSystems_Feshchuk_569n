import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Вхідні дані (варіант 10)
# --------------------------

lam = 0.042          # довжина хвилі λ, м (4.2 см)
h = 0.12             # відстань між стрижнями h, м (12 см)
l = 0.237            # довжина стрижня l, м (23.7 см)
d = 0.023            # діаметр середній d_cp = 2.3 см

eps_r = 2.2          # діелектрична проникність
xi = 1 / np.sqrt(eps_r)   # наближення для коефіцієнта уповільнення

k = 2 * np.pi / lam  # хвильове число

# Кутова сітка
theta_deg = np.linspace(0, 180, 2000)
theta = np.deg2rad(theta_deg)


# ---------------------------------------
# (4.3) — F_B(theta): ДС хвилі, що біжить
# ---------------------------------------

def F_B(theta):
    num = np.sin(np.pi * (xi - np.cos(theta)) * l / lam)
    den = np.pi * (xi - np.cos(theta))
    return np.abs(num / den)


# ---------------------------------------
# (4.4) — F_1E(theta): елементна ДС у E-площині
# ---------------------------------------

def Lambda(x):     # лямбда-функція (спрощення)
    return np.sinc(x / np.pi)

def F_1E(theta):
    return np.abs(Lambda(np.pi * d * np.sin(theta) / lam) * np.cos(theta))


# ---------------------------------------
# (4.5) — F_1H(theta): елементна ДС у H-площині
# ---------------------------------------

def F_1H(theta):
    return np.abs(Lambda(np.pi * d * np.sin(theta) / lam))


# ----------------------------------------------------
# Однострижнева ДС: F_E = F_B * F_1E ; F_H = F_B * F_1H
# ----------------------------------------------------

F_E = F_B(theta) * F_1E(theta)
F_H = F_B(theta) * F_1H(theta)

F_E_norm = F_E / np.max(F_E)
F_H_norm = F_H / np.max(F_H)


# ----------------------------------------------------
# Множник грат для двох стрижнів — F_C(theta)
# (4.18)
# ----------------------------------------------------

def F_C(theta):
    return np.cos(np.pi * h / lam * np.sin(theta))


# ----------------------------------------------------
# Двострижнева ДС у H-площині (4.25)
# ----------------------------------------------------

F2_H = F_H * F_C(theta)
F2_H_norm = F2_H / np.max(F2_H)


# ----------------------------------------------------
# Двострижнева ДС у E-площині (4.29)
# ----------------------------------------------------

F2_E = F_E * F_C(theta)
F2_E_norm = F2_E / np.max(F2_E)


# ----------------------------------------------------
# Функція для визначення HPBW (по рівню 0.707)
# ----------------------------------------------------

def HPBW(pattern, theta):
    patt = pattern / np.max(pattern)
    target = 1 / np.sqrt(2)  # 0.707

    i0 = np.argmax(patt)

    # Лівий перетин
    left = None
    for i in range(i0, 0, -1):
        if patt[i] >= target and patt[i-1] < target:
            left = theta[i]
            break

    # Правий перетин
    right = None
    for i in range(i0, len(patt)-1):
        if patt[i] >= target and patt[i+1] < target:
            right = theta[i]
            break

    if left is None or right is None:
        return None

    return np.degrees(right - left)


print("HPBW (1 стрижень, H-площина):", HPBW(F_H_norm, theta))
print("HPBW (1 стрижень, E-площина):", HPBW(F_E_norm, theta))
print("HPBW (2 стрижні, H-площина):", HPBW(F2_H_norm, theta))
print("HPBW (2 стрижні, E-площина):", HPBW(F2_E_norm, theta))


# ----------------------------------------------------
# Графіки
# ----------------------------------------------------

plt.figure(figsize=(10,6))
plt.plot(theta_deg, F_H_norm, label="1 стрижень H")
plt.plot(theta_deg, F2_H_norm, label="2 стрижні H")
plt.axhline(0.707, color='gray', linestyle='--')
plt.xlabel("θ (град)")
plt.ylabel("Нормована ДС")
plt.title("ДС у H-площині (варіант 10)")
plt.grid(True)
plt.legend()

plt.figure(figsize=(10,6))
plt.plot(theta_deg, F_E_norm, label="1 стрижень E")
plt.plot(theta_deg, F2_E_norm, label="2 стрижні E")
plt.axhline(0.707, color='gray', linestyle='--')
plt.xlabel("θ (град)")
plt.ylabel("Нормована ДС")
plt.title("ДС у E-площині (варіант 10)")
plt.grid(True)
plt.legend()

plt.show()
