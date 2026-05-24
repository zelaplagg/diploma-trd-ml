import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all
from src.preprocessor import drop_useless_sensors, add_condition_labels, normalize, get_feature_columns
from src.feature_engineering import engineer_features
from src.model import load_ensemble

# ── Завантаження даних та моделі ─────────────────────────────────────────
print("Завантаження даних та моделі...")
data = load_all('data/raw')
df = data['train']

df = drop_useless_sensors(df)
df = add_condition_labels(df)
df = engineer_features(df)
df, _ = normalize(df, df)

feature_cols = get_feature_columns(df)
X = df[feature_cols]
y = df['condition']

model = load_ensemble('models')

# ── Отримання ймовірностей ───────────────────────────────────────────────
print("Обчислення ймовірностей...")
y_score = model.predict_proba(X)

# Бінаризація міток для побудови ROC (one-vs-rest)
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = 3

class_names = {
    0: 'Нормальна робота',
    1: 'Передвідмовний стан',
    2: 'Критична деградація'
}

colors = ['#2980B9', '#27AE60', '#E74C3C']
linestyles = ['-', '--', '-.']

# ── Побудова графіка ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFAFA')

fpr_all = {}
tpr_all = {}
roc_auc_all = {}

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc_val = auc(fpr, tpr)
    fpr_all[i] = fpr
    tpr_all[i] = tpr
    roc_auc_all[i] = roc_auc_val

    ax.plot(fpr, tpr,
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=2.5,
            label=f'Клас {i}: {class_names[i]} (AUC = {roc_auc_val:.4f})')

# Macro-average ROC
all_fpr = np.unique(np.concatenate([fpr_all[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
mean_tpr /= n_classes
macro_auc = auc(all_fpr, mean_tpr)

ax.plot(all_fpr, mean_tpr,
        color='#8E44AD',
        linewidth=3,
        linestyle=':',
        label=f'Macro-average (AUC = {macro_auc:.4f})')

# Діагональна лінія (випадковий класифікатор)
ax.plot([0, 1], [0, 1],
        color='#95A5A6',
        linewidth=1.2,
        linestyle='--',
        alpha=0.7,
        label='Випадковий класифікатор (AUC = 0.5000)')

# Виділення оптимальної точки для класу 2
i_opt = 2
optimal_idx = np.argmax(tpr_all[i_opt] - fpr_all[i_opt])
ax.scatter(fpr_all[i_opt][optimal_idx],
           tpr_all[i_opt][optimal_idx],
           color=colors[i_opt],
           s=100, zorder=5, marker='*')
ax.annotate(
    f'Оптимальний поріг\n(TPR={tpr_all[i_opt][optimal_idx]:.3f}, FPR={fpr_all[i_opt][optimal_idx]:.3f})',
    xy=(fpr_all[i_opt][optimal_idx], tpr_all[i_opt][optimal_idx]),
    xytext=(0.15, 0.75),
    fontsize=9,
    color=colors[i_opt],
    arrowprops=dict(arrowstyle='->', color=colors[i_opt], lw=1.2),
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors[i_opt], alpha=0.85)
)

# Оформлення
ax.set_xlabel('False Positive Rate (FPR)', fontsize=13, labelpad=8)
ax.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=13, labelpad=8)
ax.set_title(
    'Рисунок 6.1 — ROC-криві ансамблевої моделі RF + XGBoost\n'
    'для трьох класів технічного стану ТРД (датасет NASA C-MAPSS FD001)',
    fontsize=12, fontweight='bold', pad=16
)

ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.grid(True, linestyle='--', alpha=0.4, color='#BDC3C7')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(loc='lower right', fontsize=10,
          framealpha=0.95, edgecolor='#BDC3C7', fancybox=True)

# Текстовий блок з підсумковими AUC
textstr = f'Macro AUC = {macro_auc:.4f}'
ax.text(0.98, 0.08, textstr,
        transform=ax.transAxes,
        fontsize=11, fontweight='bold',
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.4',
                  facecolor='#F8F9FA',
                  edgecolor='#8E44AD',
                  alpha=0.9))

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300,
            bbox_inches='tight', facecolor='white')
print(f"\nAUC результати:")
for i in range(n_classes):
    print(f"  Клас {i} ({class_names[i]}): AUC = {roc_auc_all[i]:.4f}")
print(f"  Macro-average AUC: {macro_auc:.4f}")
print("\nЗбережено: roc_curves.png")
plt.show()
