import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from config import *

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
IMG_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============ PARTE 2: EXPLORACIÓN ============
csv_path = sorted(RAW_DIR.glob("*.csv"))[0]
df = pd.read_csv(csv_path)
print("Archivo:", csv_path.name)
print("\n--- Primeros 10 registros ---\n", df.head(10))
print("\nDimensiones:", df.shape)
print("\nColumnas y tipos:\n", df.dtypes)
print("\nEstadísticas descriptivas:\n", df.describe(include="all").T)
print("\nValores nulos:\n", df.isnull().sum())
print("\nDuplicados:", df.duplicated().sum())

feats = pick_features(df)
print("\nVariables para K-Means:", feats)

# 3 gráficas exploratorias
df[feats].hist(bins=30, figsize=(14, 8), edgecolor="black")
plt.suptitle("Distribución de variables"); plt.tight_layout()
plt.savefig(IMG_DIR / "eda_histogramas.png", dpi=150); plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df[feats].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlación entre variables"); plt.tight_layout()
plt.savefig(IMG_DIR / "eda_correlacion.png", dpi=150); plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=(df[feats] - df[feats].mean()) / df[feats].std())
plt.xticks(rotation=30); plt.title("Boxplots (estandarizados) - valores atípicos")
plt.tight_layout(); plt.savefig(IMG_DIR / "eda_boxplots.png", dpi=150); plt.close()

# ============ PARTE 3: LIMPIEZA ============
df = df.drop_duplicates().reset_index(drop=True)
for c in df.columns:
    if not pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].astype(str).str.strip().replace("nan", np.nan)
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0])
for c in feats:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df[c] = df[c].fillna(df[c].median())

# Atípicos: regla IQR. Los valores fuera de [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
# se recortan al límite (no se borran filas, así se conservan los 50,000 registros).
for c in feats:
    q1, q3 = df[c].quantile([.25, .75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[c] < lo) | (df[c] > hi)).sum()
    print(f"{c}: {n_out} atípicos recortados")
    df[c] = df[c].clip(lo, hi)

print("\nNulos tras limpieza:", df.isnull().sum().sum(), "| Registros:", len(df))

# Escalamiento: igual que en p2
scaler = StandardScaler()
X = scaler.fit_transform(df[feats])

# ============ PARTE 4: K-MEANS ============
sample = 10000 if len(X) > 10000 else None   # silhouette con muestra para que no tarde
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    rows.append({"k": k,
                 "inercia": km.inertia_,
                 "silhouette": silhouette_score(X, km.labels_, sample_size=sample, random_state=RANDOM_STATE),
                 "davies_bouldin": davies_bouldin_score(X, km.labels_)})
metrics = pd.DataFrame(rows)
print("\nEvaluación de K:\n", metrics.round(3))
metrics.to_csv(PROCESSED_DIR / "metricas_k.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(metrics.k, metrics.inercia, "o-"); ax[0].set_title("Método del codo"); ax[0].set_xlabel("K")
ax[1].plot(metrics.k, metrics.silhouette, "o-", color="green"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("K")
plt.tight_layout(); plt.savefig(IMG_DIR / "k_codo_silhouette.png", dpi=150); plt.close()

K = K_FINAL or int(metrics.loc[metrics.silhouette.idxmax(), "k"])
K2 = K_ALT or K + 1
print(f"\nK final: {K} | K alterno: {K2}")

# Modelo final
model = KMeans(n_clusters=K, n_init=10, random_state=RANDOM_STATE)
df["cluster"] = model.fit_predict(X)

centroids = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=feats)
centroids.index.name = "cluster"
centroids_z = pd.DataFrame(model.cluster_centers_, columns=feats)
print("\nCentroides (escala original):\n", centroids.round(2))
print("\nCentroides (z-score, para interpretar):\n", centroids_z.round(2))
print("\nRegistros por clúster:\n", df.cluster.value_counts().sort_index())
centroids.round(3).to_csv(PROCESSED_DIR / "centroides.csv")

plt.figure(figsize=(9, 4))
sns.heatmap(centroids_z, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
plt.title("Centroides estandarizados"); plt.tight_layout()
plt.savefig(IMG_DIR / "centroides_heatmap.png", dpi=150); plt.close()

# Gráfica de agrupamientos: dos variables reales + centroides (como en p2)
vx, vy = feats[0], feats[1]
plt.figure(figsize=(9, 6))
plt.scatter(df[vx], df[vy], c=df["cluster"], cmap="viridis", s=8, alpha=0.4)
plt.scatter(centroids[vx], centroids[vy], color="red", marker="X", s=250, label="Centroides")
plt.title(f"Clústeres K-Means (K={K})")
plt.xlabel(vx); plt.ylabel(vy)
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(IMG_DIR / "clusters_scatter.png", dpi=150); plt.close()

# Comparación de dos configuraciones (K=4 contra K=3)
model2 = KMeans(n_clusters=K2, n_init=10, random_state=RANDOM_STATE).fit(X)
comp = pd.DataFrame({
    "config": [f"A: K={K}", f"B: K={K2}"],
    "inercia": [model.inertia_, model2.inertia_],
    "silhouette": [silhouette_score(X, model.labels_, sample_size=sample, random_state=RANDOM_STATE),
                   silhouette_score(X, model2.labels_, sample_size=sample, random_state=RANDOM_STATE)],
    "davies_bouldin": [davies_bouldin_score(X, model.labels_), davies_bouldin_score(X, model2.labels_)],
})
print("\nComparación de configuraciones:\n", comp.round(3))
comp.to_csv(PROCESSED_DIR / "comparacion_configs.csv", index=False)

# ============ PARTE 5: INTERPRETACIÓN ============
df["cluster_name"] = df.cluster.map(lambda i: CLUSTER_NAMES.get(i, f"Clúster {i}"))
stats = df.groupby("cluster")[feats + EXTRA].agg(["mean", "median", "std", "min", "max"]).round(2)
print("\nEstadísticas por clúster:\n", stats)
print("\nPerfil por clúster (variables extra):\n", df.groupby("cluster")[EXTRA].mean().round(2))
stats.to_csv(PROCESSED_DIR / "estadisticas_por_cluster.csv")

# Guardar resultados
df.to_csv(PROCESSED, index=False)
FEATURES_JSON.write_text(json.dumps({"features": feats, "extra": EXTRA, "k": K,
                                     "silhouette": float(comp.silhouette[0])}))
print("\nListo ->", PROCESSED)