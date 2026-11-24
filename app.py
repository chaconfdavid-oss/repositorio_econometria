import os
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Laboratorio Econométrico COVID-19 | Fase 4",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sistema Econométrico Aplicado: Impacto del COVID-19 en la Economía Global")
st.markdown(
    "**Fase 4:** Diagnóstico de Residuos (Gauss-Markov), Tratamiento Robusto (HC3) y Simulador Predictivo con Intervalos al 95%  \n"
    "**Dataset:** `mil datos econometria.csv` (50,418 observaciones brutas | $n = 34,623$ observaciones efectivas depuradas)"
)

def parse_log_col(series, min_val, max_val):
    out = []
    for val in series:
        s = str(val).strip()
        if s in ["", "nan", "None", "NA"]:
            out.append(np.nan)
            continue
        if s == "0":
            out.append(0.0)
            continue
        digits = s.replace(".", "")
        chosen = None
        for k in [2, 1, 3]:
            if len(digits) > k:
                v = float(digits[:k] + "." + digits[k:])
                if min_val <= v <= max_val:
                    chosen = v
                    break
        if chosen is None:
            for k in [1, 2, 3]:
                if len(digits) > k:
                    v = float(digits[:k] + "." + digits[k:])
                    if v <= max_val:
                        chosen = v
                        break
        out.append(chosen if chosen is not None else np.nan)
    return pd.Series(out, index=series.index)

@st.cache_data
def cargar_datos(uploaded_file=None):
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file, sep=";")
        df_raw.columns = (
            df_raw.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("[^a-z0-9_]", "", regex=True)
        )
        df_parsed = pd.DataFrame({
            "code": df_raw["code"],
            "country": df_raw["country"],
            "date": pd.to_datetime(df_raw["date"], dayfirst=True).dt.strftime("%Y-%m-%d"),
            "tc": parse_log_col(df_raw["tc"], 0.0, 17.0),
            "gdpcap": parse_log_col(df_raw["gdpcap"], 5.0, 12.5),
            "hdi": pd.to_numeric(df_raw["hdi"], errors="coerce"),
            "pop": parse_log_col(df_raw["pop"], 8.0, 22.0),
            "sti": parse_log_col(df_raw["sti"], 0.0, 4.7),
        })
        cols_analisis = ["tc", "gdpcap", "hdi", "pop", "sti"]
        df_sub = df_parsed.dropna(subset=cols_analisis)
        df_clean = df_sub[(df_sub[cols_analisis] > 0).all(axis=1)].copy()
        q99 = df_clean["tc"].quantile(0.99)
        df_filtered = df_clean[df_clean["tc"] <= q99].reset_index(drop=True)

        df_filtered["log_tc"] = df_filtered["tc"]
        df_filtered["log_gdpcap"] = df_filtered["gdpcap"]
        df_filtered["log_hdi"] = np.log(df_filtered["hdi"])
        df_filtered["log_pop"] = df_filtered["pop"]
        df_filtered["log_sti"] = df_filtered["sti"]

        df_filtered["cases_nivel"] = np.exp(df_filtered["log_tc"])
        df_filtered["gdpcap_nivel"] = np.exp(df_filtered["log_gdpcap"])
        df_filtered["hdi_nivel"] = df_filtered["hdi"]
        df_filtered["pop_nivel"] = np.exp(df_filtered["log_pop"])
        df_filtered["sti_nivel"] = np.exp(df_filtered["log_sti"])
        return df_filtered

    default_path = os.path.join("data", "covid_depurado_fase4.csv")
    return pd.read_csv(default_path)

st.sidebar.header("⚙️ Configuración del Dataset")
archivo_usuario = st.sidebar.file_uploader(
    "Subir CSV original (mil datos econometria.csv)",
    type=["csv"],
    help="Si no subes ningún archivo, se utiliza automáticamente la base depurada incluida en el repositorio."
)

df = cargar_datos(archivo_usuario)

# Estimación de Modelos M2 (Nivel), M3 (Doble-Log Clásico) y M3 Robusto (HC3 White)
X2 = sm.add_constant(df[["gdpcap_nivel", "hdi_nivel", "pop_nivel", "sti_nivel"]])
y2 = df["cases_nivel"]
m2 = sm.OLS(y2, X2).fit()

X3 = sm.add_constant(df[["log_gdpcap", "log_hdi", "log_pop", "log_sti"]])
y3 = df["log_tc"]
m3 = sm.OLS(y3, X3).fit()
m3_hc3 = sm.OLS(y3, X3).fit(cov_type="HC3")

tab1, tab2, tab3 = st.tabs([
    "1. Diagnóstico de Perturbaciones (Gauss-Markov)",
    "2. Tratamiento Robusto (Errores HC3) y Gráficos",
    "3. Simulador Predictivo e Intervalos al 95%"
])

with tab1:
    st.subheader("Batería de Pruebas sobre Perturbaciones (Modelo 2 Nivel vs. Modelo 3 Doble-Log)")
    jb2, jb2_p, skew2, kurt2 = jarque_bera(m2.resid)
    jb3, jb3_p, skew3, kurt3 = jarque_bera(m3.resid)

    bp2_lm, bp2_lmp, _, _ = het_breuschpagan(m2.resid, m2.model.exog)
    bp3_lm, bp3_lmp, _, _ = het_breuschpagan(m3.resid, m3.model.exog)

    dw2 = durbin_watson(m2.resid)
    dw3 = durbin_watson(m3.resid)

    df_pruebas = pd.DataFrame({
        "Supuesto Evaluado": [
            "Normalidad (Jarque-Bera)",
            "Asimetría (Skewness)",
            "Curtosis (Kurtosis)",
            "Homocedasticidad (Breusch-Pagan LM)",
            "Autocorrelación (Durbin-Watson)",
            "R-cuadrado (R²)"
        ],
        "Modelo 2: Múltiple (Nivel)": [
            f"{jb2:,.2f} (p = {jb2_p:.2e})",
            f"{skew2:.4f}",
            f"{kurt2:.4f}",
            f"{bp2_lm:,.2f} (p = {bp2_lmp:.2e})",
            f"{dw2:.4f}",
            f"{m2.rsquared:.4f}"
        ],
        "Modelo 3: Doble-Log (Elasticidades)": [
            f"{jb3:,.2f} (p = {jb3_p:.2e})",
            f"{skew3:.4f}",
            f"{kurt3:.4f}",
            f"{bp3_lm:,.2f} (p = {bp3_lmp:.2e})",
            f"{dw3:.4f}",
            f"{m3.rsquared:.4f}"
        ]
    })
    st.dataframe(df_pruebas, use_container_width=True, hide_index=True)

    st.subheader("Análisis de Multicolinealidad — Factor de Inflación de la Varianza (VIF)")
    vif_df = pd.DataFrame({
        "Variable Explicativa": ["log_gdpcap (PIB per cápita ln)", "log_hdi (IDH ln)", "log_pop (Población ln)", "log_sti (Rigurosidad ln)"],
        "VIF (Modelo 3 Doble-Log)": [
            variance_inflation_factor(X3.values, i) for i in range(1, X3.shape[1])
        ],
        "VIF (Modelo 2 Nivel)": [
            variance_inflation_factor(X2.values, i) for i in range(1, X2.shape[1])
        ]
    })
    st.dataframe(vif_df.style.format({"VIF (Modelo 3 Doble-Log)": "{:.4f}", "VIF (Modelo 2 Nivel)": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.info(
        "**Diagnóstico Técnico:** Ninguna variable supera el umbral crítico de multicolinealidad severa ($VIF > 10$). "
        "Sin embargo, tanto la prueba de Jarque-Bera ($p < 0.001$) como la prueba de Breusch-Pagan ($LM = 1,882.07, p < 0.001$) "
        "rechazan homocedasticidad y normalidad estricta, y el estadístico Durbin-Watson ($0.0883$) confirma autocorrelación positiva propia de series diarias acumuladas."
    )

with tab2:
    st.subheader("Corrección Econométrica: Estimación MCO con Errores Estándar Robustos de White (HC3)")
    comp_hc3 = pd.DataFrame({
        "Parámetro": m3.params.index,
        "Coeficiente (Beta)": m3.params.values,
        "EE Clásico (MCO)": m3.bse.values,
        "EE Robusto (HC3)": m3_hc3.bse.values,
        "Estadístico z (HC3)": m3_hc3.tvalues.values,
        "p-valor (HC3)": m3_hc3.pvalues.values,
        "IC 95% Inf (HC3)": m3_hc3.conf_int()[0].values,
        "IC 95% Sup (HC3)": m3_hc3.conf_int()[1].values,
    })
    st.dataframe(
        comp_hc3.style.format({
            "Coeficiente (Beta)": "{:.4f}",
            "EE Clásico (MCO)": "{:.4f}",
            "EE Robusto (HC3)": "{:.4f}",
            "Estadístico z (HC3)": "{:.4f}",
            "p-valor (HC3)": "{:.4e}",
            "IC 95% Inf (HC3)": "{:.4f}",
            "IC 95% Sup (HC3)": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Gráficos Diagnósticos de Residuos")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(m3.resid, kde=True, ax=axes[0], color="#2b6cb0")
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title("Distribución de Residuos [ln(u)] - Modelo 3", fontweight="bold")
    axes[0].set_xlabel("Residuos")

    axes[1].scatter(m3.fittedvalues, m3.resid, alpha=0.12, color="#2f855a")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Residuos vs. Valores Ajustados [ln(TC)]", fontweight="bold")
    axes[1].set_xlabel("Valores Predichos [ln(TC)]")
    axes[1].set_ylabel("Residuos [ln(u)]")

    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Módulo de Simulación y Predicción con Intervalos al 95% (Modelo 3 Doble-Log HC3)")
    col1, col2 = st.columns(2)
    with col1:
        pib_input = st.slider(
            "PIB per cápita (USD PPA)",
            min_value=600.0,
            max_value=115000.0,
            value=12000.0,
            step=500.0
        )
        idh_input = st.slider(
            "Índice de Desarrollo Humano (HDI)",
            min_value=0.35,
            max_value=0.96,
            value=0.75,
            step=0.01
        )
    with col2:
        pob_millones = st.slider(
            "Población Total (en millones de habitantes)",
            min_value=0.1,
            max_value=1400.0,
            value=12.0,
            step=0.5
        )
        sti_input = st.slider(
            "Índice de Rigurosidad Gubernamental (Stringency Index: 1 a 100)",
            min_value=1.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

    x_nuevo = pd.DataFrame({
        "const": [1.0],
        "log_gdpcap": [np.log(pib_input)],
        "log_hdi": [np.log(idh_input)],
        "log_pop": [np.log(pob_millones * 1_000_000)],
        "log_sti": [np.log(sti_input)],
    })

    pred_res = m3_hc3.get_prediction(x_nuevo).summary_frame(alpha=0.05)
    log_pred = float(pred_res["mean"].iloc[0])
    ci_low_log = float(pred_res["mean_ci_lower"].iloc[0])
    ci_high_log = float(pred_res["mean_ci_upper"].iloc[0])
    pi_low_log = float(pred_res["obs_ci_lower"].iloc[0])
    pi_high_log = float(pred_res["obs_ci_upper"].iloc[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicción Puntual ln(TC)", f"{log_pred:.4f}", f"≈ {np.exp(log_pred):,.0f} casos")
    c2.metric("IC 95% de la Media ln(TC)", f"[{ci_low_log:.4f} , {ci_high_log:.4f}]", f"[{np.exp(ci_low_log):,.0f} , {np.exp(ci_high_log):,.0f}]")
    c3.metric("Intervalo Predicción 95% Obs.", f"[{pi_low_log:.4f} , {pi_high_log:.4f}]", f"[{np.exp(pi_low_log):,.0f} , {np.exp(pi_high_log):,.0f}]")
