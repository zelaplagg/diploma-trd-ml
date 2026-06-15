import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import drop_useless_sensors
from src.feature_engineering import engineer_features, get_feature_columns
from src.predictor import load_artifacts, predict_condition, predict_rul, get_feature_importance_for_prediction, CLASS_NAMES, CLASS_COLORS, CLASS_EMOJI

st.set_page_config(
    page_title="TRD Diagnostics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Exo 2', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0A0E1A 0%, #0D1B2A 50%, #0A0E1A 100%);
    }
    .header-block {
        background: linear-gradient(90deg, #001F3F, #003366);
        border: 1px solid #00A8FF;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 32px;
        box-shadow: 0 0 30px rgba(0, 168, 255, 0.15);
    }
    .header-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.8rem;
        color: #00A8FF;
        letter-spacing: 3px;
        margin: 0;
        text-transform: uppercase;
    }
    .header-subtitle {
        color: #8899AA;
        font-size: 0.85rem;
        letter-spacing: 2px;
        margin-top: 6px;
        text-transform: uppercase;
    }
    .upload-zone {
        background: linear-gradient(135deg, #0D1B2A, #001525);
        border: 2px dashed #1A5080;
        border-radius: 16px;
        padding: 48px 32px;
        text-align: center;
        margin: 24px 0;
        transition: border-color 0.3s;
    }
    .upload-zone:hover { border-color: #00A8FF; }
    .upload-title {
        font-family: 'Share Tech Mono', monospace;
        color: #00A8FF;
        font-size: 1.1rem;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    .upload-hint { color: #556677; font-size: 0.85rem; letter-spacing: 1px; }
    .metric-card {
        background: linear-gradient(135deg, #0D1B2A, #001F3F);
        border: 1px solid #1A3A5C;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00A8FF;
    }
    .metric-label {
        color: #8899AA;
        font-size: 0.75rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 4px;
    }
    .status-normal {
        background: linear-gradient(135deg, #0A2A1A, #0D3B1F);
        border: 2px solid #00CC66;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 25px rgba(0,204,102,0.15);
    }
    .status-warning {
        background: linear-gradient(135deg, #2A1A00, #3B2A00);
        border: 2px solid #FFAA00;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 25px rgba(255,170,0,0.15);
    }
    .status-critical {
        background: linear-gradient(135deg, #2A0A0A, #3B0D0D);
        border: 2px solid #FF3333;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 25px rgba(255,51,51,0.2);
    }
    .status-text-normal  { font-size: 1.4rem; color: #00CC66; font-weight: 700; letter-spacing: 1px; }
    .status-text-warning { font-size: 1.4rem; color: #FFAA00; font-weight: 700; letter-spacing: 1px; }
    .status-text-critical{ font-size: 1.4rem; color: #FF3333; font-weight: 700; letter-spacing: 1px; }
    .recommendation-box {
        background: #0D1B2A;
        border-left: 4px solid #00A8FF;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 16px 0;
        color: #AABBCC;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #00A8FF44, transparent);
        margin: 24px 0;
    }
    .section-title {
        font-family: 'Share Tech Mono', monospace;
        color: #00A8FF;
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .preview-table {
        background: #0D1B2A;
        border: 1px solid #1A3A5C;
        border-radius: 8px;
        padding: 16px;
    }
    .info-badge {
        display: inline-block;
        background: #001F3F;
        border: 1px solid #00A8FF44;
        border-radius: 6px;
        padding: 4px 12px;
        color: #8899AA;
        font-size: 0.8rem;
        margin: 4px;
        font-family: 'Share Tech Mono', monospace;
    }
    /* Streamlit overrides */
    .stButton > button {
        background: linear-gradient(135deg, #003366, #0055AA) !important;
        color: #00A8FF !important;
        border: 1px solid #00A8FF !important;
        border-radius: 8px !important;
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 2px !important;
        font-size: 1rem !important;
        padding: 12px 32px !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0055AA, #0077CC) !important;
        box-shadow: 0 0 20px rgba(0,168,255,0.3) !important;
    }
    div[data-testid="stFileUploader"] {
        background: transparent !important;
    }
    .stDataFrame { background: #0D1B2A; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return load_artifacts('models')


def plot_probabilities(probabilities: dict) -> go.Figure:
    classes = list(probabilities.keys())
    values  = list(probabilities.values())
    colors  = ['#00CC66', '#FFAA00', '#FF3333']
    fig = go.Figure(go.Bar(
        x=values, y=classes, orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(color='#AABBCC', size=13)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,27,42,0.8)',
        font=dict(color='#AABBCC', family='Exo 2'),
        xaxis=dict(range=[0, 115], showgrid=True, gridcolor='#1A2A3A', ticksuffix='%', color='#8899AA'),
        yaxis=dict(color='#AABBCC', tickfont=dict(size=12)),
        margin=dict(l=10, r=60, t=10, b=10), height=180, showlegend=False
    )
    return fig


def plot_feature_importance(top_features: list) -> go.Figure:
    names  = [f['feature'] for f in reversed(top_features)]
    values = [f['importance'] for f in reversed(top_features)]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation='h',
        marker=dict(color=values, colorscale=[[0,'#003366'],[0.5,'#0066CC'],[1,'#00A8FF']], showscale=False),
        text=[f"{v:.2f}%" for v in values],
        textposition='outside',
        textfont=dict(color='#AABBCC', size=11)
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,27,42,0.8)',
        font=dict(color='#AABBCC', family='Exo 2'),
        xaxis=dict(showgrid=True, gridcolor='#1A2A3A', ticksuffix='%', color='#8899AA'),
        yaxis=dict(color='#AABBCC', tickfont=dict(size=11)),
        margin=dict(l=10, r=60, t=10, b=10), height=220, showlegend=False
    )
    return fig


def plot_sensor_trend(df: pd.DataFrame, sensor: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['cycle'], y=df[sensor], mode='lines', name=sensor.upper(),
        line=dict(color='#00A8FF', width=1.5), opacity=0.7
    ))
    window = min(10, len(df))
    smooth = df[sensor].rolling(window=window, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df['cycle'], y=smooth, mode='lines', name='Тренд',
        line=dict(color='#FF6B00', width=2.5, dash='dot')
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,27,42,0.8)',
        font=dict(color='#AABBCC', family='Exo 2'),
        xaxis=dict(title='Польотний цикл', showgrid=True, gridcolor='#1A2A3A', color='#8899AA'),
        yaxis=dict(title=sensor.upper(), showgrid=True, gridcolor='#1A2A3A', color='#8899AA'),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#AABBCC')),
        margin=dict(l=10, r=10, t=10, b=10), height=260
    )
    return fig


def run_diagnostics(df: pd.DataFrame, ensemble, scaler):
    """Повна діагностика завантажених даних."""
    df = drop_useless_sensors(df)
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)

    last_record = df.tail(1).copy()
    condition   = predict_condition(last_record, ensemble, scaler, feature_cols)
    rul_info    = predict_rul(df)
    top_features = get_feature_importance_for_prediction(ensemble, feature_cols)

    return {
        'n_cycles':    int(df['cycle'].max()),
        'condition':   condition,
        'rul':         rul_info,
        'top_features': top_features,
        'df':          df,
        'feature_cols': feature_cols
    }


def main():
    # ── Заголовок ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="header-block">
        <p class="header-title">✈ TRD Diagnostics System</p>
        <p class="header-subtitle">Підсистема діагностики технічного стану турбореактивного двигуна · NASA C-MAPSS · RF + XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Завантаження моделі ────────────────────────────────────────────────
    try:
        ensemble, scaler = load_model()
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()

    # ── Ініціалізація стану ────────────────────────────────────────────────
    if 'report' not in st.session_state:
        st.session_state.report = None
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = None

    # ── КРОК 1: Завантаження файлу ─────────────────────────────────────────
    st.markdown('<p class="section-title">① Завантаження даних двигуна</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Завантажте файл з даними польотних циклів двигуна (формат NASA C-MAPSS, .txt або .csv)",
        type=['txt', 'csv'],
        label_visibility="visible"
    )

    if uploaded is not None:
        COLUMNS = ['engine_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
                  [f's{i}' for i in range(1, 22)]
        try:
            df_raw = pd.read_csv(uploaded, sep=r'\s+', header=None, names=COLUMNS)
            st.session_state.df_raw = df_raw
            st.session_state.report = None  # скидаємо попередній звіт
        except Exception as e:
            st.error(f"❌ Помилка читання файлу: {e}")
            st.stop()

        # Превью даних
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">② Попередній огляд даних</p>', unsafe_allow_html=True)

        n_engines = df_raw['engine_id'].nunique()
        n_records = len(df_raw)
        n_cycles  = int(df_raw['cycle'].max())
        n_sensors = len([c for c in df_raw.columns if c.startswith('s')])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{n_engines}</div><div class="metric-label">Двигунів</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{n_records}</div><div class="metric-label">Записів</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{n_cycles}</div><div class="metric-label">Макс. циклів</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{n_sensors}</div><div class="metric-label">Сенсорів</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Вибір двигуна
        engine_ids = sorted(df_raw['engine_id'].unique())
        selected_engine = st.selectbox(
            "Оберіть двигун для діагностики",
            options=engine_ids,
            format_func=lambda x: f"Двигун #{x}"
        )
        st.session_state.selected_engine = selected_engine

        with st.expander("Переглянути перші рядки даних"):
            engine_preview = df_raw[df_raw['engine_id'] == selected_engine].head(10)
            st.dataframe(engine_preview, use_container_width=True)

        # ── КРОК 2: Кнопка діагностики ────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">③ Запуск діагностики</p>', unsafe_allow_html=True)

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            diagnose_btn = st.button("🔍 ЗАПУСТИТИ ДІАГНОСТИКУ", use_container_width=True)

        if diagnose_btn:
            engine_data = df_raw[df_raw['engine_id'] == selected_engine].copy()

            # Анімація аналізу
            progress_bar = st.progress(0)
            status_text  = st.empty()

            steps = [
                (15, "Нормалізація сенсорних даних..."),
                (35, "Інженерія ознак (63 параметри)..."),
                (55, "Аналіз часового ряду деградації..."),
                (75, "Класифікація RF + XGBoost Ensemble..."),
                (90, "Оцінка залишкового ресурсу (RUL)..."),
                (100, "Формування діагностичного звіту..."),
            ]
            for pct, msg in steps:
                status_text.markdown(f'<p style="color:#8899AA;font-family:Share Tech Mono,monospace;font-size:0.85rem;letter-spacing:1px;">⚙ {msg}</p>', unsafe_allow_html=True)
                progress_bar.progress(pct)
                time.sleep(0.4)

            try:
                report = run_diagnostics(engine_data, ensemble, scaler)
                report['engine_id'] = selected_engine
                st.session_state.report = report
            except Exception as e:
                st.error(f"❌ Помилка діагностики: {e}")

            progress_bar.empty()
            status_text.empty()

    # ── КРОК 3: Звіт ──────────────────────────────────────────────────────
    if st.session_state.report is not None:
        report = st.session_state.report
        c = report['condition']
        r = report['rul']
        df = report['df']

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">④ Діагностичний звіт</p>', unsafe_allow_html=True)

        # Метрики
        col1, col2, col3, col4 = st.columns(4)
        status_class = {0: 'normal', 1: 'warning', 2: 'critical'}[c['condition_class']]
        rul_val = max(0, r['rul_cycles'])

        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">#{report["engine_id"]}</div><div class="metric-label">Номер двигуна</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{report["n_cycles"]}</div><div class="metric-label">Польотних циклів</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{rul_val}</div><div class="metric-label">Залишковий ресурс</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{c["confidence"]}%</div><div class="metric-label">Впевненість моделі</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Статус + ймовірності
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown(f"""
            <div class="status-{status_class}">
                <div style="font-size:3rem">{c['condition_emoji']}</div>
                <div class="status-text-{status_class}">{c['condition_name']}</div>
                <div style="color:#8899AA;font-size:0.8rem;margin-top:8px;letter-spacing:1px;">ТЕХНІЧНИЙ СТАН ДВИГУНА</div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="recommendation-box">📋 {r["recommendation"]}</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<p class="section-title">Розподіл ймовірностей класів</p>', unsafe_allow_html=True)
            st.plotly_chart(plot_probabilities(c['probabilities']), use_container_width=True, config={'displayModeBar': False})

            st.markdown('<p class="section-title">Топ-5 діагностичних параметрів</p>', unsafe_allow_html=True)
            st.plotly_chart(plot_feature_importance(report['top_features']), use_container_width=True, config={'displayModeBar': False})

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Графіки трендів сенсорів
        st.markdown('<p class="section-title">Тренди деградації сенсорів</p>', unsafe_allow_html=True)

        engine_df = df[df['engine_id'] == report['engine_id']] if 'engine_id' in df.columns else df
        sensors = [c for c in engine_df.columns if c.startswith('s') and len(c) <= 4]

        selected_sensor = st.selectbox("Сенсор", options=sensors,
                                        index=sensors.index('s11') if 's11' in sensors else 0)

        if selected_sensor in engine_df.columns:
            st.plotly_chart(plot_sensor_trend(engine_df, selected_sensor), use_container_width=True, config={'displayModeBar': False})

        # Усі сенсори разом (grid)
        with st.expander("Переглянути всі сенсори"):
            key_sensors = ['s4', 's11', 's12', 's15', 's17', 's20', 's21']
            available = [s for s in key_sensors if s in engine_df.columns]
            cols = st.columns(2)
          for i, sensor in enumerate(available):
    with cols[i % 2]:
        st.markdown(f'<p style="color:#8899AA;font-size:0.8rem;letter-spacing:1px;">{sensor.upper()}</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_sensor_trend(engine_df, sensor), use_container_width=True, config={'displayModeBar': False}, key=f"sensor_{sensor}")
        
    elif st.session_state.df_raw is None:
        # Початковий стан — підказка
        st.markdown("""
        <div class="upload-zone">
            <p class="upload-title">📂 ЗАВАНТАЖТЕ ФАЙЛ ДАНИХ</p>
            <p class="upload-hint">Підтримувані формати: .txt (NASA C-MAPSS), .csv · Роздільник: пробіл</p>
            <p class="upload-hint" style="margin-top:8px;">Файли датасету: train_FD001.txt · test_FD001.txt</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
