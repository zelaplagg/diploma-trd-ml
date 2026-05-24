import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Додаємо кореневу папку до шляху
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_all
from src.preprocessor import drop_useless_sensors
from src.feature_engineering import engineer_features, get_feature_columns
from src.predictor import load_artifacts, diagnose_engine, CLASS_NAMES, CLASS_COLORS, CLASS_EMOJI

# ── Конфігурація сторінки ─────────────────────────────────────────────────
st.set_page_config(
    page_title="TRD Diagnostics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS стилі ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Exo 2', sans-serif;
    }

    .main { background-color: #0A0E1A; }

    .stApp {
        background: linear-gradient(135deg, #0A0E1A 0%, #0D1B2A 50%, #0A0E1A 100%);
    }

    /* Заголовок */
    .header-block {
        background: linear-gradient(90deg, #001F3F, #003366);
        border: 1px solid #00A8FF;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 0 30px rgba(0, 168, 255, 0.15);
    }

    .header-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2rem;
        color: #00A8FF;
        letter-spacing: 3px;
        margin: 0;
        text-transform: uppercase;
    }

    .header-subtitle {
        color: #8899AA;
        font-size: 0.9rem;
        letter-spacing: 2px;
        margin-top: 6px;
        text-transform: uppercase;
    }

    /* Метрики */
    .metric-card {
        background: linear-gradient(135deg, #0D1B2A, #001F3F);
        border: 1px solid #1A3A5C;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: #00A8FF;
        box-shadow: 0 0 20px rgba(0, 168, 255, 0.1);
    }

    .metric-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00A8FF;
    }

    .metric-label {
        color: #8899AA;
        font-size: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Статус картки */
    .status-normal {
        background: linear-gradient(135deg, #0A2A1A, #0D3B1F);
        border: 2px solid #00CC66;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 25px rgba(0, 204, 102, 0.15);
    }

    .status-warning {
        background: linear-gradient(135deg, #2A1A00, #3B2A00);
        border: 2px solid #FFAA00;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 25px rgba(255, 170, 0, 0.15);
    }

    .status-critical {
        background: linear-gradient(135deg, #2A0A0A, #3B0D0D);
        border: 2px solid #FF3333;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 25px rgba(255, 51, 51, 0.2);
    }

    .status-text-normal  { font-size: 1.5rem; color: #00CC66; font-weight: 700; }
    .status-text-warning { font-size: 1.5rem; color: #FFAA00; font-weight: 700; }
    .status-text-critical{ font-size: 1.5rem; color: #FF3333; font-weight: 700; }

    /* Sidebar */
    .css-1d391kg { background-color: #0D1B2A; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1B2A, #001F3F);
        border-right: 1px solid #1A3A5C;
    }

    /* Роздільник */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #00A8FF, transparent);
        margin: 20px 0;
    }

    /* Рекомендація */
    .recommendation-box {
        background: #0D1B2A;
        border-left: 4px solid #00A8FF;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 16px 0;
        color: #AABBCC;
        font-size: 0.95rem;
    }

    /* Таблиця важливості */
    .feature-row {
        display: flex;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #1A2A3A;
    }

    .stSelectbox label, .stSlider label {
        color: #8899AA !important;
        font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.85rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Завантаження даних та моделі (кешування) ─────────────────────────────
@st.cache_resource
def load_model_and_data():
    ensemble, scaler = load_artifacts('models')
    data = load_all('data/raw')
    df_test = data['test']
    df_test = drop_useless_sensors(df_test)
    df_test = engineer_features(df_test)
    feature_cols = get_feature_columns(df_test)
    return ensemble, scaler, df_test, feature_cols, data['rul']


# ── Функції для графіків ──────────────────────────────────────────────────
def plot_probabilities(probabilities: dict) -> go.Figure:
    classes = list(probabilities.keys())
    values  = list(probabilities.values())
    colors  = ['#00CC66', '#FFAA00', '#FF3333']

    fig = go.Figure(go.Bar(
        x=values, y=classes,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0)', width=0)
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(color='#AABBCC', size=13)
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,27,42,0.8)',
        font=dict(color='#AABBCC', family='Exo 2'),
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor='#1A2A3A',
            ticksuffix='%',
            color='#8899AA'
        ),
        yaxis=dict(color='#AABBCC', tickfont=dict(size=13)),
        margin=dict(l=10, r=60, t=10, b=10),
        height=180,
        showlegend=False
    )
    return fig


def plot_feature_importance(top_features: list) -> go.Figure:
    names  = [f['feature'] for f in reversed(top_features)]
    values = [f['importance'] for f in reversed(top_features)]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation='h',
        marker=dict(
            color=values,
            colorscale=[[0, '#003366'], [0.5, '#0066CC'], [1, '#00A8FF']],
            showscale=False
        ),
        text=[f"{v:.2f}%" for v in values],
        textposition='outside',
        textfont=dict(color='#AABBCC', size=11)
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,27,42,0.8)',
        font=dict(color='#AABBCC', family='Exo 2'),
        xaxis=dict(
            showgrid=True, gridcolor='#1A2A3A',
            ticksuffix='%', color='#8899AA'
        ),
        yaxis=dict(color='#AABBCC', tickfont=dict(size=11)),
        margin=dict(l=10, r=60, t=10, b=10),
        height=220,
        showlegend=False
    )
    return fig


def plot_sensor_trend(df_engine: pd.DataFrame, sensor: str) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_engine['cycle'],
        y=df_engine[sensor],
        mode='lines',
        name=sensor.upper(),
        line=dict(color='#00A8FF', width=1.5),
        opacity=0.7
    ))

    # Ковзне середнє
    window = min(10, len(df_engine))
    smooth = df_engine[sensor].rolling(window=window, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df_engine['cycle'],
        y=smooth,
        mode='lines',
        name='Тренд',
        line=dict(color='#FF6B00', width=2.5, dash='dot'),
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,27,42,0.8)',
        font=dict(color='#AABBCC', family='Exo 2'),
        xaxis=dict(
            title='Польотний цикл',
            showgrid=True, gridcolor='#1A2A3A', color='#8899AA'
        ),
        yaxis=dict(
            title=sensor.upper(),
            showgrid=True, gridcolor='#1A2A3A', color='#8899AA'
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#AABBCC')
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        height=250
    )
    return fig


# ── ГОЛОВНИЙ ІНТЕРФЕЙС ────────────────────────────────────────────────────
def main():
    # Заголовок
    st.markdown("""
    <div class="header-block">
        <p class="header-title">✈ TRD Diagnostics System</p>
        <p class="header-subtitle">Підсистема діагностики технічного стану турбореактивного двигуна · NASA C-MAPSS · RF + XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

    # Завантаження
    with st.spinner("Завантаження моделі..."):
        try:
            ensemble, scaler, df_test, feature_cols, df_rul = load_model_and_data()
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
            st.stop()

    engine_ids = sorted(df_test['engine_id'].unique())

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ ПАРАМЕТРИ")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        selected_engine = st.selectbox(
            "Номер двигуна",
            options=engine_ids,
            index=0
        )

        sensors_available = [c for c in df_test.columns if c.startswith('s') and len(c) <= 4]
        selected_sensor = st.selectbox(
            "Сенсор для графіку тренду",
            options=sensors_available,
            index=sensors_available.index('s11') if 's11' in sensors_available else 0
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 ІНФОРМАЦІЯ")
        st.markdown(f"**Двигунів у базі:** {len(engine_ids)}")
        st.markdown(f"**Ознак моделі:** {len(feature_cols)}")
        st.markdown(f"**Алгоритм:** RF + XGBoost Ensemble")
        st.markdown(f"**Датасет:** NASA C-MAPSS FD001")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        run_btn = st.button("🔍 ЗАПУСТИТИ ДІАГНОСТИКУ", use_container_width=True)

    # ── Основна панель ────────────────────────────────────────────────────
    if run_btn or True:
        report = diagnose_engine(
            selected_engine, df_test, ensemble, scaler, feature_cols
        )
        c = report['condition']
        r = report['rul']

        # Визначаємо CSS клас статусу
        status_class = {0: 'normal', 1: 'warning', 2: 'critical'}[c['condition_class']]
        status_text_class = f"status-text-{status_class}"

        # ── Рядок метрик ──────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">#{selected_engine}</div>
                <div class="metric-label">Номер двигуна</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{report['n_cycles']}</div>
                <div class="metric-label">Польотних циклів</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            rul_val = max(0, r['rul_cycles'])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{rul_val}</div>
                <div class="metric-label">Залишковий ресурс (циклів)</div>
            </div>""", unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{c['confidence']}%</div>
                <div class="metric-label">Впевненість моделі</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Статус + ймовірності ───────────────────────────────────────
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown(f"""
            <div class="status-{status_class}">
                <div style="font-size:3rem">{c['condition_emoji']}</div>
                <div class="{status_text_class}">{c['condition_name']}</div>
                <div style="color:#8899AA; font-size:0.85rem; margin-top:8px; letter-spacing:1px;">
                    ТЕХНІЧНИЙ СТАН ДВИГУНА
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="recommendation-box">
                📋 {r['recommendation']}
            </div>""", unsafe_allow_html=True)

        with col_right:
            st.markdown("#### Розподіл ймовірностей класів")
            fig_prob = plot_probabilities(c['probabilities'])
            st.plotly_chart(fig_prob, use_container_width=True, config={'displayModeBar': False})

            st.markdown("#### Топ-5 діагностичних параметрів")
            fig_fi = plot_feature_importance(report['top_features'])
            st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── Графік тренду сенсора ──────────────────────────────────────
        st.markdown(f"#### Тренд деградації сенсора {selected_sensor.upper()}")
        engine_data = df_test[df_test['engine_id'] == selected_engine]

        if selected_sensor in engine_data.columns:
            fig_trend = plot_sensor_trend(engine_data, selected_sensor)
            st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Сенсор недоступний для цього двигуна")

        # ── Еталонний RUL (якщо є) ─────────────────────────────────────
        true_rul_row = df_rul[df_rul['engine_id'] == selected_engine]
        if not true_rul_row.empty:
            true_rul = int(true_rul_row['true_RUL'].values[0])
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#00CC66">{true_rul}</div>
                    <div class="metric-label">Еталонний RUL (NASA)</div>
                </div>""", unsafe_allow_html=True)
            with col_b:
                predicted = max(0, r['rul_cycles'])
                error = abs(predicted - true_rul)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#FFAA00">{predicted}</div>
                    <div class="metric-label">Прогнозований RUL</div>
                </div>""", unsafe_allow_html=True)
            with col_c:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#FF6B00">{error}</div>
                    <div class="metric-label">Абсолютна похибка (циклів)</div>
                </div>""", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
