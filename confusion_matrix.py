import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all
from src.preprocessor import drop_useless_sensors, add_condition_labels, normalize, get_feature_columns
from src.feature_engineering import engineer_features
from src.model import load_ensemble
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ── Завантаження ──────────────────────────────────────────────────────────
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

# Розбивка 80/20
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = load_ensemble('models')

# ── Обчислення матриці помилок ────────────────────────────────────────────
print("Обчислення матриці помилок...")
y_pred = model.predict(X_val)
cm = confusion_matrix(y_val, y_pred)

class_names = [
    'Клас 0\nНормальна\nробота',
    'Клас 1\nПередвідмовний\nстан',
    'Клас 2\nКритична\nдеградація'
]

# ── Побудова теплової карти ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('white')

# ── Лівий: абсолютні значення ──────────────────────────────────────────
ax1 = axes[0]
im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

ax1.set_xticks(np.arange(len(class_names)))
ax1.set_yticks(np.arange(len(class_names)))
ax1.set_xticklabels(class_names, fontsize=10)
ax1.set_yticklabels(class_names, fontsize=10)
ax1.set_xlabel('Передбачений клас', fontsize=12, labelpad=10)
ax1.set_ylabel('Фактичний клас', fontsize=12, labelpad=10)
ax1.set_title('(а) Абсолютні значення', fontsize=12, fontweight='bold', pad=12)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = 'white' if cm[i, j] > thresh else 'black'
        ax1.text(j, i, f'{cm[i, j]}',
                 ha='center', va='center',
                 fontsize=14, fontweight='bold', color=color)

# Виділення діагоналі
for i in range(len(class_names)):
    ax1.add_patch(plt.Rectangle(
        (i - 0.5, i - 0.5), 1, 1,
        fill=False, edgecolor='#E74C3C', linewidth=2.5
    ))

# ── Правий: нормалізовані значення ─────────────────────────────────────
ax2 = axes[1]
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap='Greens', vmin=0, vmax=1)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

ax2.set_xticks(np.arange(len(class_names)))
ax2.set_yticks(np.arange(len(class_names)))
ax2.set_xticklabels(class_names, fontsize=10)
ax2.set_yticklabels(class_names, fontsize=10)
ax2.set_xlabel('Передбачений клас', fontsize=12, labelpad=10)
ax2.set_ylabel('Фактичний клас', fontsize=12, labelpad=10)
ax2.set_title('(б) Нормалізовані значення', fontsize=12, fontweight='bold', pad=12)

thresh_norm = 0.5
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        color = 'white' if cm_norm[i, j] > thresh_norm else 'black'
        ax2.text(j, i, f'{cm_norm[i, j]:.3f}',
                 ha='center', va='center',
                 fontsize=13, fontweight='bold', color=color)

for i in range(len(class_names)):
    ax2.add_patch(plt.Rectangle(
        (i - 0.5, i - 0.5), 1, 1,
        fill=False, edgecolor='#E74C3C', linewidth=2.5
    ))

# Загальний заголовок
fig.suptitle(
    'Рисунок 6.2 — Матриця помилок ансамблевої моделі RF + XGBoost\n'
    '(валідаційна вибірка 20%, датасет NASA C-MAPSS FD001)',
    fontsize=12, fontweight='bold', y=1.02
)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300,
            bbox_inches='tight', facecolor='white')

print("\nМатриця помилок:")
print(cm)
print(f"\nТочність (Accuracy): {np.trace(cm) / np.sum(cm) * 100:.2f}%")
print("Збережено: confusion_matrix.png")
plt.show()
