# Proyecto de Sistemas Econométricos — Fase 4: Diagnóstico de Residuos, Tratamiento Robusto y Dashboard Interactivo

**Dataset:** Impact of Covid-19 Pandemic on the Global Economy (`mil datos econometria.csv`)  
**Muestra Bruta:** 50,418 registros | **Muestra Efectiva Depurada ($n$):** 34,623 observaciones (210 países/territorios)

---

## 1. Informe Consolidado de Diagnósticos Econométricos (Supuestos de Gauss-Markov)

Se evaluaron las perturbaciones $\hat{u}_i$ del **Modelo 2 (Lineal Múltiple en Nivel)** y del **Modelo 3 (Doble-Log de Elasticidades)**:

$$\ln(TC_i) = \beta_0 + \beta_1 \ln(GDPCAP_i) + \beta_2 \ln(HDI_i) + \beta_3 \ln(POP_i) + \beta_4 \ln(STI_i) + u_i$$

### 1.1 Resumen de Batería de Pruebas

| Prueba / Diagnóstico | Modelo 2: Múltiple (Nivel) | Modelo 3: Doble-Log | Decisión Estadística ($\alpha = 0.05$) |
| :--- | :---: | :---: | :--- |
| **Normalidad (Jarque-Bera)** | $5,571,514.90$ ($p = 0.0000$) | $1,416.77$ ($p = 2.25 \times 10^{-308}$) | Se rechaza normalidad estricta; sin embargo, por el Teorema Central del Límite ($n = 34,623$) los estimadores son asintóticamente normales y la transformación logarítmica reduce la asimetría y curtosis ($3.162 \approx 3.0$). |
| **Homocedasticidad (Breusch-Pagan LM)** | $1,213.71$ ($p = 1.70 \times 10^{-261}$) | $1,882.07$ ($p = 0.0000$) | Se rechaza $H_0$ de varianza constante. Existe heterocedasticidad entre economías de distinto tamaño. |
| **Autocorrelación (Durbin-Watson)** | $0.1395$ | $0.0883$ | $DW < 2$ indica autocorrelación serial positiva esperada al tratarse de registros diarios acumulados por país. |
| **Multicolinealidad (VIF máximo)** | $2.42$ (`gdpcap_nivel`) | $9.80$ (`log_gdpcap`) | Todos los regresores cumplen $VIF < 10$ (ausencia de multicolinealidad severa). |

### 1.2 Tabla de Factor de Inflación de la Varianza (VIF)

| Variable Explicativa | VIF (Modelo 3 Doble-Log) | VIF (Modelo 2 Nivel) |
| :--- | :---: | :---: |
| `log_gdpcap` (PIB per cápita $\ln$) | `9.7981` | `2.4163` |
| `log_hdi` (Índice de Desarrollo Humano $\ln$) | `9.7222` | `2.3881` |
| `log_pop` (Población total $\ln$) | `1.0220` | `1.0079` |
| `log_sti` (Índice de Rigurosidad $\ln$) | `1.0092` | `1.0259` |

### 1.3 Tratamiento Econométrico Aplicado: Errores Estándar Robustos de White (HC3)

Ante la presencia confirmada de heterocedasticidad, se reestimó el **Modelo 3 (Doble-Log)** mediante la matriz de covarianzas consistente con heterocedasticidad (**HC3**):

| Parámetro | Coeficiente ($\hat{\beta}$) | EE Clásico | EE Robusto (HC3) | Estadístico $z$ | $p$-valor | IC 95% Inf | IC 95% Sup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `const` | `-13.6278` | `0.4064` | `0.3941` | `-34.577` | `0.0000` | `-14.4003` | `-12.8553` |
| `log_gdpcap` | `0.6684` | `0.0326` | `0.0322` | `20.730` | `0.0000` | `0.6052` | `0.7316` |
| `log_hdi` | `0.8214` | `0.1638` | `0.1611` | `5.099` | `0.0000` | `0.5056` | `1.1371` |
| `log_pop` | `0.7826` | `0.0076` | `0.0078` | `100.352` | `0.0000` | `0.7673` | `0.7979` |
| `log_sti` | `0.7722` | `0.0249` | `0.0357` | `21.602` | `0.0000` | `0.7021` | `0.8422` |

---

## 2. Estructura del Repositorio GitHub y Ejecución Local

```text
proyecto_econometria_covid_fase4/
├── app.py                        # Dashboard Interactivo Streamlit + Simulador IC 95%
├── requirements.txt              # Dependencias de Python
├── README.md                     # Informe consolidado de diagnósticos econométricos
└── data/
    └── covid_depurado_fase4.csv  # Dataset efectivo depurado (n = 34,623)
```

### Ejecución en local o Streamlit Community Cloud:
```bash
pip install -r requirements.txt
streamlit run app.py
```
