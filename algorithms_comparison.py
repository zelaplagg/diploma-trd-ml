import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Дані з реальних досліджень на NASA C-MAPSS ────────────────────────────
# Джерела: Alwashahi et al. (2025), DeVol et al. (2023),
#          Sun et al. (2025), Asif et al. (2022)

algorithms = ['SVM', 'Decision\nTree', 'Random\nForest', 'XGBoost', 'LSTM', 'CNN']

metrics = {
    'Accuracy (%)':   [78.4, 81.2, 95.5, 93.8, 91.2, 89.7],
    'F1-score (%)':   [74.1, 78.6, 94.8, 92.3, 90.5, 88.2],
    'ROC-AUC (%)':    [76.3, 79.4, 96.1, 94.2, 92.8, 90.5],
}

colors = {
    'Accuracy (%)':  '#2980B9',
    'F1-score (%)':  '#27AE60',
    'ROC-AUC (%)':   '#8E44AD',
}

x = np.arange(len(algorithms))
width = 0.25
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

# ── Стовпці ───────────────────────────────────────────────────────────────
for i, (metric, values) in enumerate(metrics.items()):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, values, width,
                  label=metric,
                  color=colors[metric],
                  alpha=0.85,
                  edgecolor='white',
                  linewidth=0.8)
    # Підписи значень над стовпцями
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f'{val:.1f}',
                ha='center', va='bottom',
                fontsize=8.5, color='#333333', fontweight='bold')

# ── Порогова лінія вимоги ─────────────────────────────────────────────────
ax.axhline(y=90, color='#E74C3C', linewidth=1.8,
           linestyle='--', alpha=0.8, label='Мінімальна вимога (90%)')
ax.text(5.5, 90.4, 'вимога\n90%',
        fontsize=9, color='#E74C3C',
        ha='right', va='bottom', fontweight='bold')

# ── Виділення обраних алгоритмів ──────────────────────────────────────────
ax.axvspan(1.62, 2.38, alpha=0.08, color='#27AE60')
ax.axvspan(2.62, 3.38, alpha=0.08, color='#27AE60')
ax.text(2.0, 97.5, '✓ Обрано', ha='center',
        fontsize=10, color='#27AE60', fontweight='bold')
ax.text(3.0, 97.5, '✓ Обрано', ha='center',
        fontsize=10, color='#27AE60', fontweight='bold')

# ── Оформлення ────────────────────────────────────────────────────────────
ax.set_xlabel('Алгоритм машинного навчання', fontsize=12, labelpad=10)
ax.set_ylabel('Значення метрики, %', fontsize=12, labelpad=10)
ax.set_title('Рисунок 2.2 — Порівняння алгоритмів машинного навчання\n'
             'за метриками якості на датасеті NASA C-MAPSS',
             fontsize=12, fontweight='bold', pad=16)

ax.set_xticks(x)
ax.set_xticklabels(algorithms, fontsize=11)
ax.set_ylim(65, 102)
ax.set_xlim(-0.5, len(algorithms) - 0.5)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax.grid(axis='y', linestyle='--', alpha=0.4, color='#BDC3C7')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='lower right', fontsize=10,
          framealpha=0.92, edgecolor='#BDC3C7', fancybox=True)

plt.tight_layout()
plt.savefig('algorithms_comparison.png', dpi=300,
            bbox_inches='tight', facecolor='white')
print("Збережено: algorithms_comparison.png")
plt.show()
