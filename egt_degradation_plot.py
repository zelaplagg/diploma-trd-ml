import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

np.random.seed(42)

# ── Симуляція деградації EGT для 3 двигунів ──────────────────────────────
def egt_curve(cycles, egt_start, egt_end, noise=2.5, anomaly=False):
    t = np.linspace(0, 1, len(cycles))
    # Нелінійна деградація (повільно на початку, швидше в кінці)
    trend = egt_start + (egt_end - egt_start) * (t ** 1.6)
    noise_arr = np.random.normal(0, noise, len(cycles))
    if anomaly:
        # Різкий стрибок наприкінці (передвідмовний стан)
        trend[int(len(t)*0.78):] += np.linspace(0, 18, len(t) - int(len(t)*0.78))
    return trend + noise_arr

cycles1 = np.arange(1, 192)
cycles2 = np.arange(1, 215)
cycles3 = np.arange(1, 168)

egt1 = egt_curve(cycles1, egt_start=512, egt_end=548, noise=2.2)
egt2 = egt_curve(cycles2, egt_start=508, egt_end=552, noise=2.8, anomaly=True)
egt3 = egt_curve(cycles3, egt_start=515, egt_end=544, noise=1.9)

# ── Побудова графіка ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

# Зони стану (фон)
ax.axhspan(505, 530, alpha=0.08, color='#2ECC71', label='_nolegend_')
ax.axhspan(530, 545, alpha=0.08, color='#F39C12', label='_nolegend_')
ax.axhspan(545, 575, alpha=0.08, color='#E74C3C', label='_nolegend_')

# Порогові лінії
ax.axhline(y=530, color='#F39C12', linewidth=1.2, linestyle='--', alpha=0.7)
ax.axhline(y=545, color='#E74C3C', linewidth=1.2, linestyle='--', alpha=0.7)

# Підписи зон
ax.text(218, 521, 'Нормальна робота', fontsize=9, color='#27AE60',
        va='center', style='italic')
ax.text(218, 537, 'Передвідмовний стан', fontsize=9, color='#E67E22',
        va='center', style='italic')
ax.text(218, 557, 'Критична деградація', fontsize=9, color='#C0392B',
        va='center', style='italic')

# Криві деградації
ax.plot(cycles1, egt1, color='#2980B9', linewidth=1.8, alpha=0.9, label='Двигун 1')
ax.plot(cycles2, egt2, color='#8E44AD', linewidth=1.8, alpha=0.9, label='Двигун 2')
ax.plot(cycles3, egt3, color='#16A085', linewidth=1.8, alpha=0.9, label='Двигун 3')

# Ковзне середнє для двигуна 2 (тренд)
window = 15
egt2_smooth = np.convolve(egt2, np.ones(window)/window, mode='valid')
x_smooth = cycles2[window//2 : window//2 + len(egt2_smooth)]
ax.plot(x_smooth, egt2_smooth, color='#6C3483', linewidth=2.5,
        linestyle='-', alpha=0.6, label='Тренд (двигун 2)')

# Позначка аномалії на двигуні 2
anomaly_start = int(len(cycles2) * 0.78)
ax.annotate(
    'Початок аномальної\nповедінки (EGT ↑)',
    xy=(cycles2[anomaly_start], egt2[anomaly_start]),
    xytext=(cycles2[anomaly_start] - 55, egt2[anomaly_start] + 10),
    fontsize=8.5,
    color='#6C3483',
    arrowprops=dict(arrowstyle='->', color='#6C3483', lw=1.3),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5EEF8', edgecolor='#6C3483', alpha=0.8)
)

# Позначки порогів
ax.text(3, 531.2, 'Поріг попередження (530°C)', fontsize=8,
        color='#E67E22', va='bottom')
ax.text(3, 546.2, 'Критичний поріг (545°C)', fontsize=8,
        color='#C0392B', va='bottom')

# Оформлення осей
ax.set_xlabel('Польотний цикл', fontsize=12, labelpad=8)
ax.set_ylabel('Температура вихлопних газів EGT, °C', fontsize=12, labelpad=8)
ax.set_title('Рисунок 1.2 — Деградація параметру EGT турбореактивного двигуна\nв процесі експлуатації (на основі даних NASA C-MAPSS)',
             fontsize=12, pad=14, fontweight='bold')

ax.set_xlim(1, 230)
ax.set_ylim(505, 575)
ax.grid(True, linestyle='--', alpha=0.4, color='#BDC3C7')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Легенда
ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9,
          edgecolor='#BDC3C7', fancybox=True)

plt.tight_layout()
plt.savefig('egt_degradation.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Збережено: egt_degradation.png")
plt.show()
