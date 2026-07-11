# Proyección del Crecimiento de la Población Capacitada en INFOTEP 🔮

Este repositorio contiene una aplicación interactiva desarrollada en **Streamlit** para proyectar el volumen trimestral de participantes capacitados por cada Gerencia Regional del **Instituto Nacional de Formación Técnico-Profesional (INFOTEP)** de la República Dominicana, utilizando registros históricos desde 2018 hasta 2025.

La aplicación incluye un dashboard interactivo, análisis exploratorio de datos (EDA), evaluación comparativa de 4 modelos supervisados de Machine Learning y estimaciones de capacidad regional para el año 2026.

---

## 🏗️ Estructura del Proyecto
- `app.py`: Aplicación principal de Streamlit autocontenida. Contiene la ingesta, limpieza de datos, entrenamiento dinámico de modelos de regresión y la interfaz gráfica/landing del proyecto.
- `Ejecucion por Gerencia Regional, INFOTEP, 2018 - 2025.csv`: Histórico de datos de INFOTEP por regional.
- `NotebookML.ipynb`: Cuaderno de experimentación y desarrollo de modelos de Machine Learning.
- `requirements.txt`: Dependencias de librerías Python necesarias.

---

## 🛠️ Instalación y Uso

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Lanzar la aplicación
```bash
streamlit run app.py
```

---

## 👥 Autores
- **Feibert Alirio Guzmán Pérez** - Universidad UFHEC
- **Yesenia Nuñes** - Universidad UFHEC
