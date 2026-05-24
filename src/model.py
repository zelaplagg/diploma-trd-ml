import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# Назви класів
CLASS_NAMES = {
    0: 'Нормальна робота',
    1: 'Передвідмовний стан',
    2: 'Критична деградація'
}

# Фіксований seed для відтворюваності результатів
RANDOM_STATE = 42


def build_random_forest() -> RandomForestClassifier:
    """
    Створює модель Random Forest з оптимальними гіперпараметрами
    для задачі діагностики турбореактивних двигунів.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def build_xgboost() -> XGBClassifier:
    """
    Створює модель XGBoost з оптимальними гіперпараметрами.
    """
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def build_ensemble(rf_model, xgb_model) -> VotingClassifier:
    """
    Будує ансамблеву модель (Voting Classifier) з RF та XGBoost.
    Використовує м'яке голосування (soft voting) — усереднює
    ймовірності класів від обох моделей.
    """
    return VotingClassifier(
        estimators=[
            ('random_forest', rf_model),
            ('xgboost', xgb_model)
        ],
        voting='soft',
        n_jobs=-1
    )


def train(X_train, y_train) -> tuple:
    """
    Навчає ансамблеву модель RF + XGBoost.

    Args:
        X_train: матриця ознак навчальної вибірки
        y_train: вектор міток класів

    Returns:
        (ensemble, rf_model, xgb_model) — навчені моделі
    """
    print("=" * 50)
    print("НАВЧАННЯ МОДЕЛЕЙ")
    print("=" * 50)

    rf  = build_random_forest()
    xgb = build_xgboost()
    ensemble = build_ensemble(rf, xgb)

    print("Навчання Random Forest...")
    rf.fit(X_train, y_train)
    print("Random Forest навчено ✓")

    print("Навчання XGBoost...")
    xgb.fit(X_train, y_train)
    print("XGBoost навчено ✓")

    print("Навчання ансамблю RF + XGBoost...")
    ensemble.fit(X_train, y_train)
    print("Ансамбль навчено ✓")

    print("=" * 50)
    return ensemble, rf, xgb


def cross_validate(model, X, y, k: int = 10) -> dict:
    """
    Виконує k-fold крос-валідацію моделі.

    Args:
        model: навчена модель
        X:     матриця ознак
        y:     вектор міток
        k:     кількість фолдів

    Returns:
        Словник з середніми значеннями метрик
    """
    print(f"\nКрос-валідація ({k}-fold)...")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    scores = {
        'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1),
        'f1_macro': cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1),
    }

    results = {}
    for metric, vals in scores.items():
        results[metric] = {
            'mean': vals.mean(),
            'std':  vals.std()
        }
        print(f"  {metric}: {vals.mean():.4f} ± {vals.std():.4f}")

    return results


def evaluate(model, X_test, y_test) -> dict:
    """
    Оцінює якість моделі на тестовій вибірці.

    Args:
        model:  навчена модель
        X_test: матриця ознак тестової вибірки
        y_test: еталонні мітки класів

    Returns:
        Словник з метриками якості
    """
    print("\n── Оцінювання моделі ──")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'f1_macro':  f1_score(y_test, y_pred, average='macro'),
        'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall':    recall_score(y_test, y_pred, average='macro', zero_division=0),
        'roc_auc':   roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro'),
    }

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1-macro:  {metrics['f1_macro']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nДетальний звіт:")
    print(classification_report(
        y_test, y_pred,
        target_names=list(CLASS_NAMES.values())
    ))

    print("Матриця помилок:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def get_feature_importance(model, feature_cols: list,
                            top_n: int = 15) -> pd.DataFrame:
    """
    Отримує важливість ознак з Random Forest компоненти ансамблю.

    Args:
        model:        навчена ансамблева модель
        feature_cols: список назв ознак
        top_n:        кількість топ ознак для виводу

    Returns:
        DataFrame з рейтингом важливості ознак
    """
    # Отримуємо RF з ансамблю
    rf = model.named_estimators_['random_forest']
    importances = rf.feature_importances_

    fi_df = pd.DataFrame({
        'feature':    feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print(f"\nТоп-{top_n} найважливіших діагностичних параметрів:")
    print(fi_df.head(top_n).to_string(index=False))

    return fi_df


def save_models(ensemble, rf, xgb,
                models_dir: str = 'models') -> None:
    """
    Зберігає навчені моделі на диск.
    """
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(ensemble, os.path.join(models_dir, 'ensemble.pkl'))
    joblib.dump(rf,       os.path.join(models_dir, 'random_forest.pkl'))
    joblib.dump(xgb,      os.path.join(models_dir, 'xgboost.pkl'))

    print(f"\nМоделі збережено у папку: {models_dir}/")
    print("  ensemble.pkl")
    print("  random_forest.pkl")
    print("  xgboost.pkl")


def load_ensemble(models_dir: str = 'models'):
    """
    Завантажує збережену ансамблеву модель.
    """
    path = os.path.join(models_dir, 'ensemble.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Модель не знайдено: {path}")
    model = joblib.load(path)
    print(f"Модель завантажена: {path}")
    return model


# ── Швидка перевірка ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from src.data_loader import load_all
    from src.preprocessor import (
        drop_useless_sensors, add_condition_labels,
        normalize, apply_smote, get_feature_columns
    )
    from src.feature_engineering import engineer_features
    from sklearn.model_selection import train_test_split

    print("Завантаження даних...")
    data = load_all('data/raw')
    df = data['train']

    print("\nПередобробка...")
    df = drop_useless_sensors(df)
    df = add_condition_labels(df)
    df = engineer_features(df)
    df, _ = normalize(df, df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df['condition']

    X_bal, y_bal = apply_smote(X, y)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_bal, y_bal, test_size=0.2,
        random_state=RANDOM_STATE, stratify=y_bal
    )

    ensemble, rf, xgb = train(X_tr, y_tr)
    evaluate(ensemble, X_val, y_val)
    get_feature_importance(ensemble, feature_cols)
    save_models(ensemble, rf, xgb)
