import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ЛР4. ДС диэлектрической стержневой антенны
# Вариант 10. Все данные взяты из методички
# ============================================================

# ---------- 1. Исходные данные ----------
eps_r = 2.2
tg_delta = 2e-4

lambda_cm = 4.2     # длина волны, см (таблица 3, вариант 10)
h_cm      = 12.0    # расстояние между стрежнями, см (таблица 3, вариант 10)
l_cm      = 23.7    # длина стержня, см

d_max_cm  = 2.9     # параметры стержня из задания
d_min_cm  = 1.7
d_cp_cm   = 2.3

# Переводим в метры
lam = lambda_cm / 100.0
h   = h_cm / 100.0
l   = l_cm / 100.0
d_cp = d_cp_cm / 100.0

# Коэффициент уповільнення ξ по (4.7)
xi = 1 + lam / (2 * l)

print(f"xi (коэф. уповільнення) = {xi:.4f}")

# ---------- 2. Угловая сетка ----------
# Строим только переднюю полу-плоскость, как в методичке: 0…90°
theta_deg = np.linspace(0, 90, 2001)
theta = np.deg2rad(theta_deg)


# ---------- 3. Функции ДС по методике (без λ-функции) ----------

# (4.3) — множитель бегущей волны F_B(θ)
def F_B(theta):
    """
    F_B(theta) по (4.3) с нормировкой так, чтобы F_B(0) = 1.
    """
    k = np.pi * (l / lam)
    X = xi - np.cos(theta)

    num = np.sin(k * X)
    den = X * np.sin(k * (xi - 1))

    FB = np.zeros_like(theta)
    mask = np.abs(den) > 1e-12
    FB[mask] = num[mask] / den[mask]
    FB[~mask] = 1.0  # предельный случай θ→0

    return FB

# (4.18) — множитель решётки F_C(θ) для двух стержней
def F_C(theta):
    return np.cos(np.pi * h / lam * np.sin(theta))


FB = F_B(theta)       # одностержневая, бегущая волна
FC = F_C(theta)       # решётка двух стержней

# Одностержневая ДС:
F_H1 = FB             # H-плоскость
F_E1 = FB * np.cos(theta)   # E-плоскость, (4.24)

# Двухстержневая ДС:
F_H2 = FB * FC              # H-плоскость, (4.25)
F_E2 = F_H2 * np.cos(theta) # E-плоскость, (4.29)


# ---------- 4. Нормировка ----------
def norm(x):
    return np.abs(x) / np.max(np.abs(x))


# ---------- 5. Поиск нулей и максимумов (ЧИСЛЕННО, 0…90°) ----------

def find_zeros(pattern, theta, theta_deg):
    """Нули — изменение знака."""
    zeros_idx = []
    for i in range(1, len(theta)):
        if pattern[i-1] == 0:
            zeros_idx.append(i-1)
        elif pattern[i-1] * pattern[i] < 0:
            zeros_idx.append(i)
    return theta_deg[zeros_idx], zeros_idx

def find_maxima(pattern, theta, theta_deg):
    """Локальные максимумы по смене знака производной."""
    p = norm(pattern)
    d = np.gradient(p, theta)
    max_idx = []
    for i in range(1, len(theta)-1):
        if d[i-1] > 0 and d[i+1] < 0:
            max_idx.append(i)

    # Склеиваем близкие точки (шум численного дифференцирования)
    merged = []
    for idx in max_idx:
        if not merged or theta_deg[idx] - theta_deg[merged[-1]] > 0.2:
            merged.append(idx)

    return theta_deg[merged], merged


# Нули и максимумы для одностержневой F_B (H-плоскость)
zeros_FB_deg, zeros_FB_idx = find_zeros(FB, theta, theta_deg)
max_FB_deg, max_FB_idx = find_maxima(FB, theta, theta_deg)

print("\nНули F_B(θ) в диапазоне 0–90°:")
for i, ang in enumerate(zeros_FB_deg, start=1):
    print(f"  θ0{i} ≈ {ang:.2f}°")

print("\nМаксимумы F_B(θ) (боковые лепестки) в диапазоне 0–90°:")
for i, ang in enumerate(max_FB_deg, start=1):
    print(f"  θm{i} ≈ {ang:.2f}°")

# Для отчёта будем подписывать ПЕРВЫЕ 3 максимума (3 боковых лепестка)
max_FB_deg_show = max_FB_deg[:3]
max_FB_idx_show = max_FB_idx[:3]


# Нули и максимумы для решётки F_C(θ) (двухстержневая)
zeros_FC_deg, zeros_FC_idx = find_zeros(FC, theta, theta_deg)
max_FC_deg, max_FC_idx = find_maxima(FC, theta, theta_deg)

print("\nНули F_C(θ) (решётка) 0–90°:")
for i, ang in enumerate(zeros_FC_deg, start=1):
    print(f"  θ0C{i} ≈ {ang:.2f}°")

print("\nМаксимумы F_C(θ) 0–90°:")
for i, ang in enumerate(max_FC_deg, start=1):
    print(f"  θmC{i} ≈ {ang:.2f}°")


# ---------- 6. Ширина главного лепестка по формулам (4.22), (4.23) ----------

# По нулям (2θ0° ≈ 115° * sqrt(λ / l))
width_null = 115.0 * np.sqrt(lambda_cm / l_cm)
# По половине мощности (2θ0.5 ≈ 61° * sqrt(λ / l))
width_hp = 61.0 * np.sqrt(lambda_cm / l_cm)

print(f"\nШирина основного лепестка по нулям (2θ0) ≈ {width_null:.2f}°")
print(f"Ширина по половине мощности (2θ0.5) ≈ {width_hp:.2f}°")


# ============================================================
#                  7. ГРАФИКИ
# ============================================================

level_0707 = 1 / np.sqrt(2)

# --- 7.1 Одностержневая ДС: H и E на одном графике ---
plt.figure(figsize=(9, 6))
plt.plot(theta_deg, norm(FB), 'k--', label='|F_B(θ)| (H-площина)')
plt.plot(theta_deg, norm(np.cos(theta)), 'g-', label='|cos θ|')
plt.plot(theta_deg, norm(F_E1), 'b-', label='|F_E(θ)| = |F_B·cosθ|')
plt.axhline(level_0707, color='gray', linestyle='--', label='Рівень 0.707')

# подписи НУЛЕЙ F_B (возьмём первые 6 в 0–90°)
for i, idx in enumerate(zeros_FB_idx[:6], start=1):
    x = theta_deg[idx]
    y = norm(FB)[idx]
    plt.scatter(x, y, color='r')
    plt.text(x, y - 0.06, f"θ0{i}", ha='center', fontsize=8)

# подписи ПЕРВЫХ 3 максимумов боковых лепестков
for i, idx in enumerate(max_FB_idx_show, start=1):
    x = theta_deg[idx]
    y = norm(FB)[idx]
    plt.scatter(x, y, color='k', marker='x')
    plt.text(x, y + 0.04, f"θm{i}", ha='center', fontsize=8)

plt.title("Однострижнева ДС у площинах H та E (варіант 10)")
plt.xlabel("θ, град")
plt.ylabel("Нормована амплітуда")
plt.grid(True)
plt.legend()
plt.xlim(0, 90)
plt.ylim(0, 1.1)

# --- 7.2 Двухстержневая ДС — только H-плоскость ---
plt.figure(figsize=(9, 6))
plt.plot(theta_deg, norm(F_H2), 'b-', label='|F_H(θ)| для 2 стрижнів')
plt.plot(theta_deg, norm(FB), 'k--', label='|F_B(θ)| (один стрижень)')
plt.plot(theta_deg, norm(FC), 'g-', label='|F_C(θ)| (множник грат)')
plt.axhline(level_0707, color='gray', linestyle='--', label='Рівень 0.707')

# подписи нулей решётки F_C
for i, idx in enumerate(zeros_FC_idx, start=1):
    x = theta_deg[idx]
    y = norm(FC)[idx]
    plt.scatter(x, y, color='r')
    plt.text(x, y + 0.04, f"θ0C{i}", ha='center', color='r', fontsize=8)

plt.title("Двострижнева ДС у H-площині (варіант 10)")
plt.xlabel("θ, град")
plt.ylabel("Нормована амплітуда")
plt.grid(True)
plt.legend()
plt.xlim(0, 90)
plt.ylim(0, 1.1)

# --- 7.3 Двухстержневая ДС — E-плоскость ---
plt.figure(figsize=(9,6))

# Множник бегущей волны (один стержень) — пунктир (как в методичке)
plt.plot(theta_deg, norm(FB), 'k--', label='|F_B(θ)| (один стрижень)')

# Множник решётки (ОБЯЗАТЕЛЬНО)
plt.plot(theta_deg, norm(FC), 'g-', linewidth=2, label='|F_C(θ)| (множник грат)')

# ДС двострижневої в H-площині (F_B * F_C)
plt.plot(theta_deg, norm(F_H2), 'b-', label='|F_H(θ)| = |F_B·F_C|')

# Функция cosθ
plt.plot(theta_deg, norm(np.cos(theta)), 'm-', label='|cos θ|')

# Итоговая ДС в E-плоскости
plt.plot(theta_deg, norm(F_E2), 'r-', linewidth=2,
         label='|F_E(θ)| = |F_B·F_C·cosθ|')

# Уровень 0.707
plt.axhline(1/np.sqrt(2), color='gray', linestyle='--', label='Рівень 0.707')

plt.title("Двострижнева ДС у E-площині (з урахуванням множника грат)")
plt.xlabel("θ, град")
plt.ylabel("Нормована амплітуда")
plt.grid(True)
plt.legend()
plt.xlim(0, 90)
plt.ylim(0, 1.1)

plt.show()

