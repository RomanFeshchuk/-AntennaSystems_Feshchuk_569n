import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# ВАРІАНТ 10 (додані твої значення)
# ---------------------------------------------------------------------------
lam = 0.029   # довжина хвилі, м
D = 0.7       # діаметр дзеркала, м
f = 0.3       # фокусна відстань, м

# ---------------------------------------------------------------------------
# РОЗРАХУНКИ ОСНОВНИХ ПАРАМЕТРІВ
# ---------------------------------------------------------------------------
k = 2 * np.pi / lam          # хвильове число
R0 = D / 2                   # радіус дзеркала
p = 2 * f                    # подвоєний фокус

theta_deg = np.linspace(-90, 90, 3000)
theta = np.radians(theta_deg)

u = k * R0 * np.sin(theta)
v = k * p * np.sin(theta / 2)

# захист від ділення на нуль
u[u == 0] = 1e-9
v[v == 0] = 1e-9

# ---------------------------------------------------------------------------
# ФУНКЦІЇ ДС В Е- ТА Н-ПЛОЩИНАХ
# ---------------------------------------------------------------------------

def FH(theta, u, v):
    term1 = 0.74 * (v*jv(1, v)*jv(0, u) - u*jv(1, u)*jv(0, v)) / (v**2 - u**2)
    term2 = 0.26 * (jv(1, u) / u)
    term3 = -0.25 * (u*jv(1, u)*jv(2, 1.5*v) - 1.5*v*jv(1, 1.5*v)*jv(2, u)) / ((1.5*v)**2 - u**2)
    denom = 1 / (0.74*(jv(1, v) / v) + 0.13)
    return np.cos(theta/2)**2 * (term1 + term2 + term3) * denom


def FE(theta, u, v):
    term1 = 0.74 * (v*jv(1, v)*jv(0, u) - u*jv(1, u)*jv(0, v)) / (v**2 - u**2)
    term2 = 0.26 * (jv(1, u) / u)
    term3 = +0.25 * (u*jv(1, u)*jv(2, 1.5*v) - 1.5*v*jv(1, 1.5*v)*jv(2, u)) / ((1.5*v)**2 - u**2)
    denom = 1 / (0.74*(jv(1, v) / v) + 0.13)
    return np.cos(theta/2)**2 * (term1 + term2 + term3) * denom

# ---------------------------------------------------------------------------
# ОБЧИСЛЕННЯ ДС
# ---------------------------------------------------------------------------
FH_raw = FH(theta, u, v)
FE_raw = FE(theta, u, v)

# нормалізація
FH_n = FH_raw / np.max(FH_raw)
FE_n = FE_raw / np.max(FE_raw)

# ---------------------------------------------------------------------------
# ШИРИНА ГОЛОВНОЇ ПЕЛЮСТКИ (0.707)
# ---------------------------------------------------------------------------
def beamwidth(theta_deg, pattern):
    idx = np.where(pattern >= 0.707)[0]
    return theta_deg[idx[-1]] - theta_deg[idx[0]]

BW_H = beamwidth(theta_deg, FH_n)
BW_E = beamwidth(theta_deg, FE_n)

print("Ширина головної пелюстки (Н-площина):", BW_H, "град")
print("Ширина головної пелюстки (Е-площина):", BW_E, "град")

# ---------------------------------------------------------------------------
# АНАЛІЗ БОКОВИХ ПЕЛЮСТОК (SLL)
# ---------------------------------------------------------------------------
# Н-площина
main_lobe_H = np.argmax(FH_n)
peaks_H, _ = find_peaks(FH_n)
side_H = [p for p in peaks_H if abs(p - main_lobe_H) > 50]

sll_H = np.max(FH_n[side_H])
sll_H_dB = 20*np.log10(sll_H)
sll_H_angle = theta_deg[side_H][np.argmax(FH_n[side_H])]

print("\nАНАЛІЗ БОКОВИХ ПЕЛЮСТОК — Н-площина")
print("SLL =", round(sll_H_dB, 2), "дБ")
print("Кут появи =", round(sll_H_angle, 2), "град")

# Е-площина
main_lobe_E = np.argmax(FE_n)
peaks_E, _ = find_peaks(FE_n)
side_E = [p for p in peaks_E if abs(p - main_lobe_E) > 50]

sll_E = np.max(FE_n[side_E])
sll_E_dB = 20*np.log10(sll_E)
sll_E_angle = theta_deg[side_E][np.argmax(FE_n[side_E])]

print("\nАНАЛІЗ БОКОВИХ ПЕЛЮСТОК — Е-площина")
print("SLL =", round(sll_E_dB, 2), "дБ")
print("Кут появи =", round(sll_E_angle, 2), "град")

# ---------------------------------------------------------------------------
# ГРАФІКИ
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(theta_deg, FH_n)
plt.title("ДС Дзеркальної Антени — Н-площина (нормована)")
plt.xlabel("θ, градуси")
plt.ylabel("F_H(θ)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(theta_deg, FE_n)
plt.title("ДС Дзеркальної Антени — Е-площина (нормована)")
plt.xlabel("θ, градуси")
plt.ylabel("F_E(θ)")
plt.grid(True)
plt.show()
