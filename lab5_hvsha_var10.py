import numpy as np
import matplotlib.pyplot as plt
import math

# -------------------- ВХІДНІ ДАНІ --------------------
lam = 2.9   # довжина хвилі (см)
a = 2.3     # ширина широкої стінки хвилеводу (см)
lam_c = 2 * a
lam_g = lam / np.sqrt(1 - (lam / lam_c)**2)

d = 2.0     # відстань між щілинами
N = 14      # кількість щілин
nu = 1      # шаховий порядок → ν = 1

theta_deg = np.linspace(-90, 90, 20001)
theta = np.radians(theta_deg)

# -------------------- ЕЛЕМЕНТНА ДС --------------------
def F1H(theta):
    cos_t = np.cos(theta)
    num = np.cos(0.5 * np.pi * np.sin(theta))
    F = np.zeros_like(theta)

    mask = np.abs(cos_t) > 1e-6
    F[mask] = num[mask] / cos_t[mask]
    F[~mask] = 0.0
    return F

# -------------------- МНОЖНИК РЕШІТКИ --------------------
def F_HC(theta):
    psi = 2 * np.pi * d * (np.sin(theta) / lam - 1 / lam_g) + nu * np.pi
    psi2 = psi / 2
    num = np.sin(N * psi2)
    den = N * np.sin(psi2)

    F = np.zeros_like(theta)
    mask = np.abs(den) > 1e-10
    F[mask] = num[mask] / den[mask]
    F[~mask] = 1.0
    return F

# -------------------- ПОВНА ДС --------------------
F1 = np.abs(F1H(theta))
FHC = np.abs(F_HC(theta))
FH = F1 * FHC
FHn = FH / np.max(FH)

F1n = F1 / np.max(F1)
FHCn = FHC / np.max(FHC)

# -------------------- ПОШУК НУЛІВ --------------------
threshold = 0.02
is_min = (FHn[1:-1] < FHn[:-2]) & (FHn[1:-1] < FHn[2:])
is_small = FHn[1:-1] < threshold
zero_idx = np.where(is_min & is_small)[0] + 1
zero_points = np.sort(theta_deg[zero_idx])

# -------------------- ВИЗНАЧАЄМО МЕЖІ ГОЛОВНОЇ ПЕЛЮСТКИ --------------------
left_zero = zero_points[zero_points < 0][-1]   # найближчий нуль зліва
right_zero = zero_points[zero_points > 0][0]   # найближчий нуль справа

# -------------------- ПОШУК ТІЛЬКИ БОКОВИХ МАКСИМУМІВ --------------------
max_points = []
for i in range(1, len(FHn) - 1):

    if FHn[i] > FHn[i - 1] and FHn[i] > FHn[i + 1]:

        th = theta_deg[i]

        # Відсікаємо ВСІ максимуми всередині головної пелюстки:
        if left_zero < th < right_zero:
            continue

        max_points.append((th, FHn[i]))

# -------------------- ТОЧКИ РІВНЯ 0.707 --------------------
level = 0.707
crossings = []
for i in range(len(FHn) - 1):
    if (FHn[i] - level) * (FHn[i + 1] - level) < 0:
        crossings.append(theta_deg[i])

# -------------------- ГРАФІК --------------------
plt.figure(figsize=(14, 7))

plt.plot(theta_deg, F1n, label="Елементна ДС F1H(θ)", linewidth=2, color="goldenrod")
plt.plot(theta_deg, FHCn, label="Множник решітки FHC(θ)", linestyle="--", color="skyblue")
plt.plot(theta_deg, FHn, label="Повна ДС F_H(θ)", linewidth=2, color="teal")

plt.axhline(level, linestyle="--", color="orange", label="Рівень 0.707")

# Рівень 0.707
for x in crossings:
    plt.plot(x, level, 'ro')
    plt.text(x, level + 0.03, f"{x:.2f}°", ha='center', color='red')

# Нулі (лише номери)
for i, x in enumerate(zero_points):
    plt.plot(x, 0, 'kx')
    plt.text(x, -0.03, f"θ₀{i+1}", ha='center', fontsize=8)

# Бокові максимуми
for i, (ang, val) in enumerate(max_points):
    plt.plot(ang, val, 'go')
    plt.text(ang, val + 0.03, f"θₘ{i+1}={ang:.1f}°", ha='center', color='green', fontsize=9)

plt.xlabel("θ, градуси")
plt.ylabel("Нормована амплітуда")
plt.title("Повна ДС ХвЩА з підписами нулів і бокових максимумів (варіант 10)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# -------------------- ТАБЛИЦІ --------------------

print("\nТаблиця 1 — Нульові кути")
print("---------------------------------")
print("| № |   θ₀ [°]  |")
for i, ang in enumerate(zero_points):
    print(f"| {i+1:2d} | {ang:8.2f} |")
print("---------------------------------\n")

print("Таблиця 3 — Максимальні кути бокових пелюсток")
print("-------------------------------------------")
print("| № |  θₘ [°]  |  F_H(θₘ) |")
for i, (ang, val) in enumerate(max_points):
    print(f"| {i+1:2d} | {ang:8.2f} |  {val:.4f} |")
print("-------------------------------------------")

plt.show()