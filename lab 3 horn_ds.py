import numpy as np
import matplotlib.pyplot as plt
import math

# --- ВХОДНЫЕ ДАННЫЕ (Вариант 10 - Рупорная Антенна) ---
VARIANTS_NUMBER = 10
LAMBD_cm = 3.4  # Длина волны λ [см]
Ap_cm = 13.0  # Размер раскрыва в плоскости H (ap) [см]
Bp_cm = 13.0  # Размер раскрыва в плоскости E (bp) [см]
# --------------------------------------------------------------------

# Конвертация в метры и расчет констант
LAMBD = LAMBD_cm / 100.0
Ap = Ap_cm / 100.0
Bp = Bp_cm / 100.0
LEVEL_0707 = 0.707  # Уровень половинной мощности

# --- ИНИЦИАЛИЗАЦИЯ ---
FC = [1];
FE = [1];
steps = [0]
SGP1 = 0;
SGP2 = 0;
fS1 = 0;
fS2 = 0
max_y_FC = [];
max_y_FE = []

# --- ЦИКЛ РАСЧЕТА ДН (0 до 90 градусов) ---
for teta in np.arange(0.0001, np.pi / 2, 0.00001):

    sin_teta = np.sin(teta);
    cos_teta = np.cos(teta)

    # 1. ДН в плоскости H (F_H) - Нормированная Sinc-функция
    # Формула: | sin( (pi * Ap / lambda) * sin(teta) ) / ( (pi * Ap / lambda) * sin(teta) ) |
    arg_H = np.pi * Ap / LAMBD * sin_teta
    mn1 = abs(np.sin(arg_H) / arg_H) if arg_H != 0 else 1.0

    # 2. ДН в плоскости E (F_E) - Нормированная функция (учитывает косинусную зависимость)
    # Формула: | (cos( (pi * Bp / lambda) * sin(teta) ) * cos(teta)) / (1 - (2 * Bp / (pi * lambda) * sin(teta)) ** 2) |
    arg_E = np.pi * Bp / LAMBD * sin_teta
    numer = np.cos(arg_E) * cos_teta
    denom = 1.0 - (2 * Bp / (np.pi * LAMBD) * sin_teta) ** 2
    mn2 = abs(numer / denom) if abs(denom) > 1e-6 else 0.0

    FC.append(mn1);
    FE.append(mn2)

    # Оценка ШГЛ (на уровне 0.707)
    if SGP1 == 0 and LEVEL_0707 < mn1 < LEVEL_0707 + 0.0001:
        SGP1 = 2 * math.degrees(teta);
        fS1 = mn1
    if SGP2 == 0 and LEVEL_0707 < mn2 < LEVEL_0707 + 0.0001:
        SGP2 = 2 * math.degrees(teta);
        fS2 = mn2

    steps.append(math.degrees(teta))

# --- АНАЛИЗ МАКСИМУМОВ (РБП) ---
for i in range(1, len(FC) - 1):
    # F_H (Плоскость H):
    if FC[i] > FC[i - 1] and FC[i] > FC[i + 1] and steps[i] > 1.0:  # Исключаем главный максимум
        max_y_FC.append(FC[i])
    # F_E (Плоскость E):
    if FE[i] > FE[i - 1] and FE[i] > FE[i + 1] and steps[i] > 1.0:
        max_y_FE.append(FE[i])

# Вывод результатов для оценки
max_h_val = max_y_FC[0] if len(max_y_FC) > 0 else 0
max_e_val = max_y_FE[0] if len(max_y_FE) > 0 else 0

print("--- АНАЛІЗ ДС РУПОРНОЇ АНТЕНИ (Вар. 10) ---")
print(f"Ширина головної пелюстки в пл. H (F_H) = {SGP1:.2f}\u00b0")
print(f"Ширина головної пелюстки в пл. E (F_E) = {SGP2:.2f}\u00b0")
print(f"Макс. рівень 1-го Біч. Леп. (H) = {max_h_val:.3f} ({20 * np.log10(max_h_val):.2f} дБ)")
print(f"Макс. рівень 1-го Біч. Леп. (E) = {max_e_val:.3f} ({20 * np.log10(max_e_val):.2f} дБ)")

# --- ГРАФИЧЕСКАЯ ЧАСТЬ (ПОСТРОЕНИЕ) ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(steps, FC, linewidth=1.5, label="$ F_{H}(\\theta) $ (Плоскость H)")
ax.plot(steps, FE, linewidth=1.5, label="$ F_{E}(\\theta) $ (Плоскость E)")

# Пометки ШГЛ
ax.plot(SGP1 / 2, fS1, 'ro', markersize=6, label=f"ШГП H ({SGP1:.2f}\u00b0)")
ax.plot(SGP2 / 2, fS2, 'go', markersize=6, label=f"ШГП E ({SGP2:.2f}\u00b0)")

# Линия 0.707
ax.hlines(y=LEVEL_0707, xmin=0, xmax=90, colors='gray', linestyles='dotted', linewidth=1)

ax.set_title(f"Нормовані ДН Рупорної Антени (Вар. {VARIANTS_NUMBER}: {Ap_cm}x{Bp_cm}см)", fontsize=10)
ax.set_xlabel('Угол от оси антенны, $\\theta$ (\u00b0)', fontsize=10)
ax.set_ylabel('Нормована ДН, $|F(\\theta)|$', fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.set_xlim(0, 90.5)

plt.legend(loc="upper right", fontsize=8)
plt.grid(which='both', linestyle='--', linewidth=0.2, color='gray')

# Сохранение графика
fig.savefig(f"ДС_Рупорна_Вар{VARIANTS_NUMBER}_{Ap_cm}x{Bp_cm}.jpg", dpi=600)
plt.show()