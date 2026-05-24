import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# Датчики з нульовою або мінімальною варіативністю — виключаємо
# (визначено на основі аналізу датасету NASA C-MAPSS FD001)
DROP_SENSORS = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']

# Порогові значення RUL для розмічення класів
RUL_NORMAL    = 125   # > 125 циклів — нормальна робота
RUL_WARNING   = 50    # 50-125 циклів — передвідмовний стан
                      # < 50 циклів  — критична деградація

# Назви класів
CLASS_NAMES = {
    0: 'Нормальна робота',
    1: 'Передвідмовний стан',
    2: 'Критична деградація'
}


def drop_useless_sensors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Видаляє датчики що не змінюються протягом деградації
    і не несуть діагностичної інформації.
    """
    cols_to_drop = [c for c in DROP_SENSORS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Видалено {len(cols_to_drop)} неінформативних датчиків: {cols_to_drop}")
    return df


def label_condition(rul: float) -> int:
    """
    Перетворює числове значення RUL на клас технічного стану.

    Args:
        rul: залишковий ресурс у циклах

    Returns:
        0 — нормальна робота
        1 — передвідмовний стан
        2 — критична деградація
    """
    if rul > RUL_NORMAL:
        return 0
    elif rul > RUL_WARNING:
        return 1
    else:
        return 2


def add_condition_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає колонку condition_label на основі RUL.
    """
    df = df.copy()
    df['condition'] = df['RUL'].apply(label_condition)
    print(f"Розмічення класів:")
    for cls, name in CLASS_NAMES.items():
        count = (df['condition'] == cls).sum()
        pct = count / len(df) * 100
        print(f"  Клас {cls} ({name}): {count} записів ({pct:.1f}%)")
    return df


def normalize(df_train: pd.DataFrame,
              df_test: pd.DataFrame = None,
              scaler_path: str = 'models/scaler.pkl'):
    """
    Нормалізує сенсорні дані методом Min-Max до діапазону [0, 1].
    Навчає scaler на тренувальних даних та застосовує до тестових.

    Args:
        df_train: навчальна вибірка
        df_test:  тестова вибірка (опційно)
        scaler_path: шлях для збереження scaler

    Returns:
        Нормалізований df_train (та df_test якщо переданий)
    """
    # Визначаємо сенсорні колонки для нормалізації
    sensor_cols = [c for c in df_train.columns
                   if c.startswith('s') and c not in DROP_SENSORS]

    scaler = MinMaxScaler()
    df_train = df_train.copy()
    df_train[sensor_cols] = scaler.fit_transform(df_train[sensor_cols])

    # Зберігаємо scaler для подальшого використання
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler збережено: {scaler_path}")
    print(f"Нормалізовано {len(sensor_cols)} сенсорних колонок")

    if df_test is not None:
        df_test = df_test.copy()
        df_test[sensor_cols] = scaler.transform(df_test[sensor_cols])
        return df_train, df_test

    return df_train


def apply_smote(X: pd.DataFrame,
                y: pd.Series,
                random_state: int = 42):
    """
    Балансує класи методом SMOTE (Synthetic Minority Over-sampling).
    Застосовується лише до навчальної вибірки.

    Args:
        X: матриця ознак
        y: вектор міток класів
        random_state: seed для відтворюваності

    Returns:
        X_resampled, y_resampled — збалансовані дані
    """
    print(f"\nРозподіл до SMOTE: {dict(y.value_counts().sort_index())}")

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)

    print(f"Розподіл після SMOTE: {dict(pd.Series(y_res).value_counts().sort_index())}")
    return X_res, y_res


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Повертає список колонок що використовуються як ознаки для моделі.
    Виключає службові колонки.
    """
    exclude = ['engine_id', 'cycle', 'RUL', 'condition']
    return [c for c in df.columns if c not in exclude]


def preprocess_train(df: pd.DataFrame) -> tuple:
    """
    Повний пайплайн передобробки навчальних даних:
    1. Видалення неінформативних датчиків
    2. Розмічення класів
    3. Нормалізація
    4. Балансування SMOTE

    Args:
        df: навчальна вибірка з RUL

    Returns:
        X_train, y_train, feature_cols
    """
    print("=" * 50)
    print("ПЕРЕДОБРОБКА НАВЧАЛЬНИХ ДАНИХ")
    print("=" * 50)

    df = drop_useless_sensors(df)
    df = add_condition_labels(df)
    df = normalize(df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df['condition']

    X_balanced, y_balanced = apply_smote(X, y)

    print(f"\nФінальний розмір навчальної вибірки: {X_balanced.shape}")
    print("=" * 50)

    return X_balanced, y_balanced, feature_cols


def preprocess_test(df: pd.DataFrame,
                    scaler_path: str = 'models/scaler.pkl') -> tuple:
    """
    Передобробка тестових даних з використанням збереженого scaler.

    Args:
        df: тестова вибірка
        scaler_path: шлях до збереженого scaler

    Returns:
        X_test, feature_cols
    """
    print("=" * 50)
    print("ПЕРЕДОБРОБКА ТЕСТОВИХ ДАНИХ")
    print("=" * 50)

    df = drop_useless_sensors(df)

    sensor_cols = [c for c in df.columns
                   if c.startswith('s') and c not in DROP_SENSORS]

    scaler = joblib.load(scaler_path)
    df = df.copy()
    df[sensor_cols] = scaler.transform(df[sensor_cols])
    print(f"Scaler застосовано з: {scaler_path}")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]

    print(f"Розмір тестової вибірки: {X.shape}")
    print("=" * 50)

    return X, feature_cols


# ── Швидка перевірка ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from src.data_loader import load_all

    data = load_all('data/raw')

    X_train, y_train, features = preprocess_train(data['train'])
    print(f"\nОзнаки моделі ({len(features)}):")
    print(features)

    X_test, _ = preprocess_test(data['test'])
    print(f"\nТестова вибірка готова: {X_test.shape}")
