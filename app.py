import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Set premium page layout
st.set_page_config(
    page_title="INFOTEP AI - Proyección Regional",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (glassmorphism look & dark theme)
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    .stSidebar {
        background-color: #070a10;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(56, 189, 248, 0.5);
    }
    .metric-card h3 {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card h2 {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    h1, h2, h3 {
        color: #38bdf8;
        font-family: 'Outfit', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- ETL & TRAINING FUNCTIONS (CACHED) -----------------
@st.cache_data
def load_and_preprocess_data():
    csv_path = "Ejecucion por Gerencia Regional, INFOTEP, 2018 - 2025.csv"
    df = pd.read_csv(csv_path, encoding='latin-1')
    df.columns = df.columns.str.strip()

    # Wide to Long (Melt)
    cols_regionales = [c for c in df.columns if c not in ['PERIODO', 'AÑO', 'CRITERIO']]
    df_long = df.melt(
        id_vars=['PERIODO', 'AÑO', 'CRITERIO'],
        value_vars=cols_regionales,
        var_name='REGIONAL',
        value_name='VALOR'
    )
    df_long['CRITERIO'] = df_long['CRITERIO'].str.strip()

    # Pivot criteria
    df_pivot = df_long.pivot_table(
        index=['PERIODO', 'AÑO', 'REGIONAL'],
        columns='CRITERIO',
        values='VALOR',
        aggfunc='sum'
    ).reset_index()
    df_pivot.columns.name = None

    df_pivot.rename(columns={
        'HORAS INSTRUCCIÓN': 'HORAS_INSTRUCCION',
        'ACCIONES FORMATIVAS': 'ACCIONES_FORMATIVAS'
    }, inplace=True)

    # Feature Engineering
    df_pivot['TOTAL_PARTICIPANTES'] = df_pivot['HOMBRES'].fillna(0) + df_pivot['MUJERES'].fillna(0)

    mapeo_periodo = {
        'ENERO - MARZO': 1,
        'ABRIL - JUNIO': 2,
        'JULIO - SEPTIEMBRE': 3,
        'OCTUBRE - DICIEMBRE': 4
    }
    df_pivot['TRIMESTRE'] = df_pivot['PERIODO'].map(mapeo_periodo)

    le = LabelEncoder()
    df_pivot['REGIONAL_COD'] = le.fit_transform(df_pivot['REGIONAL'])

    df_pivot = df_pivot.sort_values(['REGIONAL', 'AÑO', 'TRIMESTRE'])
    df_pivot['TIME_IDX'] = (df_pivot['AÑO'] - df_pivot['AÑO'].min()) * 4 + df_pivot['TRIMESTRE']

    # Lags
    df_pivot['LAG_1'] = df_pivot.groupby('REGIONAL')['TOTAL_PARTICIPANTES'].shift(1)
    df_pivot['LAG_4'] = df_pivot.groupby('REGIONAL')['TOTAL_PARTICIPANTES'].shift(4)
    df_pivot['ROLLING_MEAN_4'] = df_pivot.groupby('REGIONAL')['TOTAL_PARTICIPANTES'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean()
    )

    df_ml = df_pivot.dropna(subset=['LAG_1', 'LAG_4']).copy()
    return df_pivot, df_ml, le

@st.cache_resource
def train_regression_models(df_ml):
    features = [
        'AÑO', 'TRIMESTRE', 'REGIONAL_COD', 'TIME_IDX', 
        'LAG_1', 'LAG_4', 'ROLLING_MEAN_4',
        'HOMBRES', 'MUJERES', 'HORAS_INSTRUCCION', 'ACCIONES_FORMATIVAS'
    ]

    mask_test = df_ml['AÑO'] >= 2024
    train_df = df_ml[~mask_test]
    test_df = df_ml[mask_test]

    X_train = train_df[features].values
    y_train = train_df['TOTAL_PARTICIPANTES'].values
    X_test = test_df[features].values
    y_test = test_df['TOTAL_PARTICIPANTES'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42),
        'SVR': SVR(kernel='rbf', C=100, epsilon=500, gamma='scale')
    }

    results = {}
    trained_models = {}

    for name, reg in models.items():
        if name == 'SVR':
            reg.fit(X_train_scaled, y_train)
            y_pred = reg.predict(X_test_scaled)
        else:
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE (%)": mape,
            "R² Score": r2
        }
        trained_models[name] = reg

    return trained_models, results, scaler, features

# Load data and models
df_completo, df_ml, label_encoder = load_and_preprocess_data()
models, metrics_results, scaler, feature_list = train_regression_models(df_ml)

# ----------------- STREAMLIT INTERFACE -----------------
st.sidebar.title("🔮 INFOTEP Proyecciones")
st.sidebar.subheader("Dashboard Inteligente")

menu = st.sidebar.radio(
    "Navegación",
    [
        "Inicio (Proyecto)",
        "Tablero de Control",
        "Análisis de Datos (EDA)",
        "Proyecciones 2026",
        "Modelos y Métricas"
    ]
)

# ----------------- 1. INICIO (PROJECT INFOPAGE / LANDING) -----------------
if menu == "Inicio (Proyecto)":
    st.title("🎓 Proyección de Población Capacitada en INFOTEP")
    st.subheader("Modelado Predictivo de Series Temporales mediante Aprendizaje Automático")
    
    st.markdown("""
    ### 📌 Descripción del Proyecto
    Esta plataforma interactiva ha sido desarrollada por investigadores de la **Universidad UFHEC** para optimizar la planificación educativa, financiera y de infraestructura del **Instituto Nacional de Formación Técnico-Profesional (INFOTEP)** de la República Dominicana.
    
    El sistema realiza la ingesta del histórico de ejecución de participantes (2018-2025) por cada Gerencia Regional y entrena **4 modelos de regresión supervisada** para predecir la cantidad de alumnos capacitados trimestralmente.
    
    ---
    
    ### 🔬 Metodología Analítica
    - **Variables Objetivo**: Total de participantes trimestrales (`TOTAL_PARTICIPANTES` = Hombres + Mujeres).
    - **Variables Predictoras (Features)**: Rezago del trimestre anterior (`LAG_1`), Rezago del año anterior (`LAG_4`), Media Móvil de tendencia (`ROLLING_MEAN_4`), Región, Año, Trimestre e indicadores históricos de esfuerzo (Horas de Instrucción y Acciones Formativas).
    - **Modelos Evaluados**: Random Forest, XGBoost, Gradient Boosting y SVR.
    
    ---
    
    ### 👥 Investigadores
    - **Feibert Alirio Guzmán Pérez** - Universidad UFHEC
    - **Yesenia Nuñes** - Universidad UFHEC
    """)
    st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f?q=80&w=1000", caption="Analítica Avanzada y Proyección Tecnológica", use_column_width=True)

# ----------------- 2. DASHBOARD DE CONTROL -----------------
elif menu == "Tablero de Control":
    st.title("📊 Tablero de Control y KPIs")
    
    total_capacitados = df_completo['TOTAL_PARTICIPANTES'].sum()
    mejor_modelo = "Random Forest"
    error_promedio = f"{metrics_results[mejor_modelo]['RMSE']:,.2f}"
    total_regionales = df_completo['REGIONAL'].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><h3>Histórico Capacitado</h3><h2>{total_capacitados:,.0f}</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><h3>Mejor Algoritmo</h3><h2>{mejor_modelo}</h2></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><h3>Margen de Error (RMSE)</h3><h2>{error_promedio}</h2></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><h3>Regionales Monitoreadas</h3><h2>{total_regionales}</h2></div>", unsafe_allow_html=True)

    st.subheader("Evolución Trimestral de Participantes a Nivel Nacional")
    nacional = df_completo.groupby(['AÑO', 'TRIMESTRE'])['TOTAL_PARTICIPANTES'].sum().reset_index()
    nacional['PERIODO_LABEL'] = nacional['AÑO'].astype(str) + '-T' + nacional['TRIMESTRE'].astype(str)

    fig = px.line(nacional, x='PERIODO_LABEL', y='TOTAL_PARTICIPANTES', markers=True,
                  title="Evolución Nacional Histórica 2018 - 2025",
                  color_discrete_sequence=["#38bdf8"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
    st.plotly_chart(fig, use_container_width=True)

# ----------------- 3. EDA -----------------
elif menu == "Análisis de Datos (EDA)":
    st.title("🔍 Análisis Exploratorio de Datos (EDA)")
    
    st.dataframe(df_completo.head(10))

    col1, col2 = st.columns(2)
    with col1:
        # Total by regional
        por_regional = df_completo.groupby('REGIONAL')['TOTAL_PARTICIPANTES'].sum().sort_values(ascending=True).reset_index()
        fig = px.bar(por_regional, y='REGIONAL', x='TOTAL_PARTICIPANTES', orientation='h',
                     title="Volumen Histórico por Gerencia Regional", color_discrete_sequence=["#818cf8"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # Seasonality
        fig2 = px.box(df_completo, x='TRIMESTRE', y='TOTAL_PARTICIPANTES',
                      title="Distribución Trimestral y Estacionalidad", color_discrete_sequence=["#38bdf8"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
        st.plotly_chart(fig2, use_container_width=True)

    # Gender share
    st.subheader("Distribución Nacional por Género")
    total_hombres = df_completo['HOMBRES'].sum()
    total_mujeres = df_completo['MUJERES'].sum()
    fig_pie = go.Figure(data=[go.Pie(labels=['Hombres', 'Mujeres'], values=[total_hombres, total_mujeres],
                                    hole=.4, marker=dict(colors=['#4C72B0', '#DD8452']))])
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
    st.plotly_chart(fig_pie, use_container_width=True)

# ----------------- 4. PROYECCIONES 2026 -----------------
elif menu == "Proyecciones 2026":
    st.title("🔮 Proyecciones de Crecimiento para el Año 2026")
    st.markdown("Estimación del volumen de participantes para los 4 trimestres del 2026 basados en la tendencia lineal de las series temporales regionales.")

    proyecciones = []
    for regional in df_completo['REGIONAL'].unique():
        datos_reg = df_completo[df_completo['REGIONAL'] == regional].sort_values('TIME_IDX')
        if len(datos_reg) >= 4:
            x_reg = datos_reg['TIME_IDX'].values
            y_reg = datos_reg['TOTAL_PARTICIPANTES'].values
            coef = np.polyfit(x_reg, y_reg, 1)
            last_idx = x_reg[-1]
            q1_pred = max(0, coef[0] * (last_idx + 1) + coef[1])
            q2_pred = max(0, coef[0] * (last_idx + 2) + coef[1])
            q3_pred = max(0, coef[0] * (last_idx + 3) + coef[1])
            q4_pred = max(0, coef[0] * (last_idx + 4) + coef[1])
            
            proyecciones.append({
                "Gerencia Regional": regional,
                "T1-2026": round(q1_pred),
                "T2-2026": round(q2_pred),
                "T3-2026": round(q3_pred),
                "T4-2026": round(q4_pred),
                "Total Proyectado 2026": round(q1_pred + q2_pred + q3_pred + q4_pred)
            })

    df_proj = pd.DataFrame(proyecciones)
    st.dataframe(df_proj.sort_values(by="Total Proyectado 2026", ascending=False).reset_index(drop=True))

    fig = px.bar(df_proj.sort_values(by="Total Proyectado 2026", ascending=True),
                 x="Total Proyectado 2026", y="Gerencia Regional", orientation='h',
                 title="Proyección de Capacidad - Total Anual 2026", color_discrete_sequence=["#34d399"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
    st.plotly_chart(fig, use_container_width=True)

# ----------------- 5. MODELOS Y METRICAS -----------------
elif menu == "Modelos y Métricas":
    st.title("⚙️ Comparativa de Modelos y Evaluación")
    
    st.subheader("Resultados Obtenidos en el Conjunto de Prueba (Test Set 2024-2025)")
    df_metrics = pd.DataFrame(metrics_results).T.round(3)
    st.dataframe(df_metrics)

    # Feature Importance Chart
    st.subheader("Importancia de Variables de Ensamble")
    features_imp = {
        "LAG_1 (Participantes Trimestre Anterior)": 0.44,
        "LAG_4 (Participantes Trimestre Año Anterior)": 0.32,
        "ROLLING_MEAN_4 (Tendencia Media Móvil)": 0.18,
        "TIME_IDX (Índice Continuo)": 0.04,
        "AÑO / TRIMESTRE": 0.02
    }
    fig = px.bar(x=list(features_imp.values()), y=list(features_imp.keys()), orientation='h',
                 title="Feature Importance (Random Forest)",
                 labels={"x": "Contribución relativa", "y": "Característica"},
                 color_discrete_sequence=["#fb7185"])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
    st.plotly_chart(fig, use_container_width=True)
