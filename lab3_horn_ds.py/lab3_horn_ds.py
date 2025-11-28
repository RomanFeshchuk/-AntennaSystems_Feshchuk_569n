import numpy as np
import matplotlib.pyplot as plt

# --- ВХОДНЫЕ ДАННЫЕ (Вариант 10 - Рупорная Антена) ---
lam = 0.034   # длина волны, м (3.4 см)
ap = 0.13     # размер раскрыва в H-плоскости, м (13 см)
bp = 0.13     # размер раскрыва в E-плоскости, м (13 см)

k = 2 * np.pi / lam
level = 0.707  # уровень -3 дБ

# Диапазон углов
theta_deg = np.linspace(0, 90, 50000)
theta = np.deg2rad(theta_deg)

# --- НОРМАЛИЗОВАННЫЕ ДН ---
# Плоскость H
argH = 0.5 * k * ap * np.sin(theta)
FH = np.ones_like(argH)
maskH = np.abs(argH) > 1e-9
FH[maskH] = np.abs(np.sin(argH[maskH]) / argH[maskH])
FH = FH / FH.max()

# Плоскость E
argE = 0.5 * k * bp * np.sin(theta)
FE = np.ones_like(argE)
maskE = np.abs(argE) > 1e-9
FE[maskE] = np.abs(np.sin(argE[maskE]) / argE[maskE] * np.cos(theta[maskE]))
FE = FE / FE.max()

# --- ПОИСК ШИРИНЫ ГЛАВНОЙ ЛЕПЕСТКИ ---
def get_hpbw(curve):
    idx = np.where(curve <= level)[0]
    if len(idx) == 0:
        return None
    edge = theta_deg[idx[0]]
    return 2 * edge, edge

hpbw_H, edge_H = get_hpbw(FH)
hpbw_E, edge_E = get_hpbw(FE)

# --- ПОИСК УРОВНЯ 1-го БОКОВОГО ЛЕПЕСТКА ---
def get_first_sidelobe(curve):
    zeros = np.where(curve < 0.01)[0]
    if len(zeros) < 2:
        return 0
    start = zeros[1]
    for i in range(start+1, len(curve)-1):
        if curve[i] > curve[i-1] and curve[i] > curve[i+1]:
            return curve[i]
    return 0

SL_H = get_first_sidelobe(FH)
SL_E = get_first_sidelobe(FE)

print("\n--- АНАЛІЗ ДС РУПОРНОЇ АНТЕНИ (Вар. 10) ---")
print(f"ШГП H = {hpbw_H:.2f}°")
print(f"ШГП E = {hpbw_E:.2f}°")
print(f"1-й бок. лепесток H = {SL_H:.3f} ({20*np.log10(SL_H):.2f} дБ)")
print(f"1-й бок. лепесток E = {SL_E:.3f} ({20*np.log10(SL_E):.2f} дБ)")

# --- ВИЗУАЛИЗАЦИЯ ---
plt.figure(figsize=(10,6))
plt.plot(theta_deg, FH, label='F_H(θ) (H-пл.)', linewidth=1.3)
plt.plot(theta_deg, FE, label='F_E(θ) (E-пл.)', linewidth=1.3)

plt.plot(edge_H, level, 'ro', label=f'ШГП H ≈ {hpbw_H:.2f}°')
plt.plot(edge_E, level, 'go', label=f'ШГП E ≈ {hpbw_E:.2f}°')

plt.axhline(level, linestyle='dashed', color='orange')
plt.grid(True, linestyle='--', linewidth=0.4)
plt.xlim(0, 90)
plt.ylim(0, 1.05)

plt.title('Нормовані ДС пірамідальної рупорної антени (варіант 10, 13×13 см)')
plt.xlabel('Кут від осі антени, θ (°)')
plt.ylabel('Нормована ДС, |F(θ)|')
plt.legend(fontsize=9)

plt.tight_layout()
plt.show()
