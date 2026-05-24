import pandas as pd
import numpy as np
import joblib
import os

# Назви класів технічного стану
CLASS_NAMES = {
    0: 'Нормальна робота',
    1: 'Передвідмовний стан',
    2: 'Критична деградація'
}

# Кольори для інтерфейсу
CLASS_COLORS = {
    0: 'green',
    1: 'orange',
    2: 'red'
}

# Emoji для інтерфейсу
CLASS_EMOJI = {
    0: '✅',
    1: '⚠️',
    2: '🔴'
}


def load_artifacts(models_dir: str = 'models') -> tuple:
    """
    Завантажує збережену модель та scaler.

    Args:
        models_dir: папка з моделями

    Returns:
        (ensemble, scaler) — модель та нормалізатор
    """
    ensemble_path = os.path.join(models_dir, 'ensemble.pkl')
    scaler_path   = os.path.join(models_dir, 'scaler.pkl')

    if not os.path.exists(ensemble_path):
        raise FileNotFoundError(f"Модель не знайдено: {ensemble_path}\nСпочатку запустіть src/model.py")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler не знайдено: {scaler_path}\nСпочатку запустіть src/model.py")

    ensemble = joblib.load(ensemble_path)
    scaler   = joblib.load(scaler_path)

    return ensemble, scaler


def predict_condition(engine_data: pd.DataFrame,
                      ensemble,
                      scaler,
                      feature_cols: list) -> dict:
    """
    Діагностує технічний стан двигуна на основі вхідних параметрів.

    Args:
        engine_data:  DataFrame з параметрами одного двигуна
        ensemble:     навчена ансамблева модель
        scaler:       нормалізатор даних
        feature_cols: список ознак що використовувались при навчанні

    Returns:
        Словник з результатами діагностики
    """
    # Нормалізація сенсорних даних
    sensor_cols = [c for c in engine_data.columns if c.startswith('s')]
    engine_data = engine_data.copy()
    engine_data[sensor_cols] = scaler.transform(engine_data[sensor_cols])

    # Отримання вектора ознак
    X = engine_data[feature_cols]

    # Прогноз класу та ймовірностей
    pred_class = ensemble.predict(X)[0]
    pred_proba = ensemble.predict_proba(X)[0]

    return {
        'condition_class': int(pred_class),
        'condition_name':  CLASS_NAMES[pred_class],
        'condition_color': CLASS_COLORS[pred_class],
        'condition_emoji': CLASS_EMOJI[pred_class],
        'probabilities': {
            CLASS_NAMES[i]: round(float(p) * 100, 2)
            for i, p in enumerate(pred_proba)
        },
        'confidence': round(float(max(pred_proba)) * 100, 2)
    }


def predict_rul(engine_data: pd.DataFrame) -> dict:
    """
    Оцінює залишковий ресурс двигуна на основі тренду деградації.
    Використовує лінійну екстраполяцію нормалізованого циклу.

    Args:
        engine_data: DataFrame з параметрами двигуна (часовий ряд)

    Returns:
        Словник з оцінкою RUL
    """
    n_cycles = len(engine_data)
    current_cycle = engine_data['cycle'].max()

    # Середній темп деградації на основі EGT proxy (s11 — найближчий до EGT)
    if 's11' in engine_data.columns and n_cycles > 5:
        s11_values = engine_data['s11'].values
        trend = np.polyfit(range(len(s11_values)), s11_values, 1)[0]
        # Груба оцінка RUL на основі тренду
        if trend > 0:
            rul_estimate = max(0, int(200 - current_cycle * (1 + trend * 10)))
        else:
            rul_estimate = max(0, int(300 - current_cycle))
    else:
        rul_estimate = max(0, int(200 - current_cycle))

    # Визначення рекомендації
    if rul_estimate > 125:
        recommendation = "Двигун у нормальному стані. Планове обслуговування не потрібне."
        urgency = "low"
    elif rul_estimate > 50:
        recommendation = "Рекомендується запланувати технічне обслуговування."
        urgency = "medium"
    else:
        recommendation = "Необхідне негайне технічне обслуговування!"
        urgency = "high"

    return {
        'rul_cycles':     rul_estimate,
        'current_cycle':  int(current_cycle),
        'recommendation': recommendation,
        'urgency':        urgency
    }


def get_feature_importance_for_prediction(ensemble,
                                           feature_cols: list,
                                           top_n: int = 5) -> list:
    """
    Повертає топ-N найважливіших параметрів що впливають на діагноз.

    Args:
        ensemble:     навчена ансамблева модель
        feature_cols: список назв ознак
        top_n:        кількість топ параметрів

    Returns:
        Список словників з назвою та важливістю параметра
    """
    rf = ensemble.named_estimators_['random_forest']
    importances = rf.feature_importances_

    fi = sorted(
        zip(feature_cols, importances),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return [
        {'feature': name, 'importance': round(float(imp) * 100, 2)}
        for name, imp in fi
    ]


def diagnose_engine(engine_id: int,
                    test_df: pd.DataFrame,
                    ensemble,
                    scaler,
                    feature_cols: list) -> dict:
    """
    Повна діагностика одного двигуна з тестової вибірки.

    Args:
        engine_id:    номер двигуна
        test_df:      тестова вибірка
        ensemble:     навчена модель
        scaler:       нормалізатор
        feature_cols: список ознак

    Returns:
        Повний діагностичний звіт
    """
    engine_data = test_df[test_df['engine_id'] == engine_id].copy()

    if engine_data.empty:
        raise ValueError(f"Двигун #{engine_id} не знайдено у тестовій вибірці")

    # Беремо останній запис для класифікації (поточний стан)
    last_record = engine_data.tail(1).copy()

    condition = predict_condition(last_record, ensemble, scaler, feature_cols)
    rul_info  = predict_rul(engine_data)
    top_features = get_feature_importance_for_prediction(ensemble, feature_cols)

    return {
        'engine_id':    engine_id,
        'n_cycles':     int(engine_data['cycle'].max()),
        'condition':    condition,
        'rul':          rul_info,
        'top_features': top_features
    }


def print_report(report: dict) -> None:
    """
    Виводить діагностичний звіт у форматованому вигляді.
    """
    print("\n" + "=" * 55)
    print(f"  ДІАГНОСТИЧНИЙ ЗВІТ — Двигун #{report['engine_id']}")
    print("=" * 55)
    print(f"  Польотних циклів: {report['n_cycles']}")

    c = report['condition']
    print(f"\n  {c['condition_emoji']} Стан: {c['condition_name']}")
    print(f"  Впевненість моделі: {c['confidence']}%")

    print("\n  Ймовірності класів:")
    for cls, prob in c['probabilities'].items():
        bar = '█' * int(prob / 5)
        print(f"    {cls:<25} {prob:>6.2f}%  {bar}")

    r = report['rul']
    print(f"\n  Залишковий ресурс (RUL): ~{r['rul_cycles']} циклів")
    print(f"  Рекомендація: {r['recommendation']}")

    print("\n  Топ-5 діагностичних параметрів:")
    for i, f in enumerate(report['top_features'], 1):
        print(f"    {i}. {f['feature']:<30} {f['importance']:.2f}%")

    print("=" * 55)


# ── Швидка перевірка ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from src.data_loader import load_all
    from src.preprocessor import drop_useless_sensors
    from src.feature_engineering import engineer_features, get_feature_columns

    print("Завантаження моделі та даних...")
    ensemble, scaler = load_artifacts('models')
    data = load_all('data/raw')

    df_test = data['test']
    df_test = drop_useless_sensors(df_test)
    df_test = engineer_features(df_test)
    feature_cols = get_feature_columns(df_test)

    # Діагностика двигунів 1, 5, 10
    for engine_id in [1, 5, 10]:
        report = diagnose_engine(engine_id, df_test, ensemble, scaler, feature_cols)
        print_report(report)
