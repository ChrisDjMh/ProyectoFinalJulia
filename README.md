# Segmentación de usuarios de streaming con K-Means

Proyecto de minería de datos: identificación de perfiles de oyentes en una plataforma de streaming musical mediante K-Means y un dashboard interactivo en Streamlit.

## Descripción
Se segmentan 50,000 usuarios según sus hábitos de escucha (horas semanales, skips diarios, playlists creadas y calificación de las recomendaciones) para proponer estrategias de retención y recomendación. Se identificaron 4 perfiles: **Críticos**, **Skippers**, **Playlists Creadors** y **Casuales**.

## Dataset
- Fuente: [Spotify User Behavior and Pattern (Kaggle)](https://www.kaggle.com/datasets/sahilislam007/spotify-user-behavior-and-pattern)
- 50,000 registros y 18 variables
- Original: `data/raw/spotify.csv` | Procesado con clústeres: `data/processed/spotify_clusters.csv`

## Herramientas
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Plotly.

## Estructura del repositorio
```
├── data/
│   ├── raw/spotify.csv
│   └── processed/        # CSV con clústeres, centroides y métricas
├── src/
│   ├── config.py         # variables, K y nombres de clústeres
│   ├── analisis_kmeans.py  # exploración, limpieza, K-Means e interpretación
│   └── app.py            # dashboard Streamlit
├── images/               # gráficas generadas
├── docs/                 # documentación en PDF
├── requirements.txt
└── README.md
```

## Cómo ejecutar
```bash
pip install -r requirements.txt
python src/analisis_kmeans.py          # genera data/processed/ e images/
python -m streamlit run src/app.py     # abre el dashboard en http://localhost:8501
```

## Resultados principales
| Clúster | Usuarios | Horas/sem | Skips/día | Playlists | Rating |
|---|---|---|---|---|---|
| Críticos | 11,207 (22.4%) | 10.58 | 9.50 | 7.52 | 2.13 |
| Skippers | 12,688 (25.4%) | 11.69 | 13.11 | 6.88 | 4.08 |
| Playlists Creadors | 11,860 (23.7%) | 10.66 | 9.36 | 11.39 | 3.98 |
| Casuales | 14,245 (28.5%) | 7.44 | 8.20 | 6.54 | 4.16 |

## Dashboard
Dos páginas: (1) Análisis general y (2) Segmentación K-Means, con KPIs, gráficas, tabla de estadísticas, tres filtros interactivos, descripción de grupos, hallazgos y recomendaciones.

## Autor
[ Morales Higuera Christian De Jesus ] - [ IDGS 9-3 ] - [ Extracción de Conocimiento en Bases de Datos ]