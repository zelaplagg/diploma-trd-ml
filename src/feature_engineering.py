import pandas as pd
import numpy as np

# Сенсори що використовуються після видалення неінформативних
INFORMATIVE_SENSORS = [
    's2', 's3', 's4', 's7', 's8', 's9',
    's11', 's12', 's13', 's14', 's15',
    's17', 's20', 's21'
]

# Розмір ковзного вікна для статистичних ознак
WINDOW_SIZE = 10


def add_rolling_features(df: pd.DataFrame,
                          sensors: list = INFORMATIVE_SENSORS,
                          window: int = WINDOW_SIZE) -> pd.DataFrame:
    """
    Додає ковзні статистики для кожного сенсора:
    - ковзне середнє (trend)
    - ковзне стандартне відхилення (volatility)

    Args:
        df:      DataFrame з даними двигунів
        sensors: список сенсорів для обробки
        window:  розмір ковзного вікна

    Returns:
        DataFrame з доданими ознаками
    """
    df = df.copy()
    new_cols = []

    for engine_id in df['engine_id'].unique():
        mask = df['engine_id'] == engine_id
        engine_data = df.loc[mask, sensors]

        for sensor in sensors:
            col_mean = f'{sensor}_mean_{window}'
            col_std  = f'{sensor}_std_{window}'

            df.loc[mask, col_mean] = (
                engine_data[sensor]
                .rolling(window=window, min_periods=1)
                .mean()
                .values
            )
            df.loc[mask, col_std] = (
                engine_data[sensor]
                .rolling(window=window, min_periods=1)
                .std()
                .fillna(0)
                .values
            )

            if col_mean not in new_cols:
                new_cols.append(col_mean)
            if col_std not in new_cols:
                new_cols.append(col_std)

    print(f"Додано {len(new_cols)} ковзних ознак (вікно={window})")
    return df


def add_degradation_rate(df: pd.DataFrame,
                          sensors: list = INFORMATIVE_SENSORS) -> pd.DataFrame:
    """
    Додає швидкість зміни (різницю між поточним та попереднім значенням)
    для ключових сенсорів — характеризує темп деградації.

    Args:
        df:      DataFrame з даними двигунів
        sensors: список сенсорів

    Returns:
        DataFrame з доданими ознаками швидкості зміни
    """
    df = df.copy()
    new_cols = []

    for engine_id in df['engine_id'].unique():
        mask = df['engine_id'] == engine_id

        for sensor in sensors:
            col_diff = f'{sensor}_diff'
            df.loc[mask, col_diff] = (
                df.loc[mask, sensor]
                .diff()
                .fillna(0)
                .values
            )
            if col_diff not in new_cols:
                new_cols.append(col_diff)

    print(f"Додано {len(new_cols)} ознак швидкості деградації")
    return df


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає ознаки на основі польотного циклу:
    - нормалізований цикл відносно максимального для двигуна
    - логарифм циклу (підсилює ранні ознаки деградації)

    Args:
        df: DataFrame з колонкою cycle та engine_id

    Returns:
        DataFrame з доданими цикловими ознаками
    """
    df = df.copy()

    max_cycles = df.groupby('engine_id')['cycle'].transform('max')
    df['cycle_normalized'] = df['cycle'] / max_cycles
    df['cycle_log'] = np.log1p(df['cycle'])

    print("Додано 2 циклові ознаки: cycle_normalized, cycle_log")
    return df


def engineer_features(df: pd.DataFrame,
                       add_rolling: bool = True,
                       add_diff: bool = True,
                       add_cycle: bool = True) -> pd.DataFrame:
    """
    Повний пайплайн інженерії ознак.

    Args:
        df:           вхідний DataFrame
        add_rolling:  додавати ковзні статистики
        add_diff:     додавати швидкість зміни
        add_cycle:    додавати циклові ознаки

    Returns:
        DataFrame з усіма інженерними ознаками
    """
    print("=" * 50)
    print("ІНЖЕНЕРІЯ ОЗНАК")
    print("=" * 50)
    print(f"Початкова кількість ознак: {len(df.columns)}")

    sensors = [s for s in INFORMATIVE_SENSORS if s in df.columns]

    if add_rolling:
        df = add_rolling_features(df, sensors)

    if add_diff:
        df = add_degradation_rate(df, sensors)

    if add_cycle:
        df = add_cycle_features(df)

    print(f"Фінальна кількість ознак: {len(df.columns)}")
    print("=" * 50)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Повертає список колонок що використовуються як ознаки.
    Виключає лише службові колонки.
    """
    exclude = {'engine_id', 'cycle', 'RUL', 'condition'}
    return [c for c in df.columns if c not in exclude]


# ── Швидка перевірка ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from src.data_loader import load_all
    from src.preprocessor import drop_useless_sensors, add_condition_labels, normalize

    data = load_all('data/raw')
    df = data['train']

    df = drop_useless_sensors(df)
    df = add_condition_labels(df)
    df, _ = normalize(df, df)
    df = engineer_features(df)

    feature_cols = get_feature_columns(df)
    print(f"\nВсього ознак для моделі: {len(feature_cols)}")
    print("Перші 10 ознак:", feature_cols[:10])
    print("\nПриклад даних:")
    print(df[feature_cols[:5]].head())
