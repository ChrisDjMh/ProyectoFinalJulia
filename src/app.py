import json
import pandas as pd
import streamlit as st
import plotly.express as px
from config import PROCESSED, FEATURES_JSON, CLUSTER_NAMES

st.set_page_config(page_title="Perfiles de Oyentes en Streaming", layout="wide")


@st.cache_data
def load():
    return pd.read_csv(PROCESSED), json.loads(FEATURES_JSON.read_text())


df, meta = load()
feats, K = meta["features"] + meta.get("extra", []), meta["k"]

# ---------- Filtros (3) ----------
st.sidebar.header("Filtros")
names = sorted(df.cluster_name.unique())
sel_clusters = st.sidebar.multiselect("Clúster", names, default=names)

cat_cols = [c for c in df.select_dtypes(exclude="number").columns
            if c != "cluster_name" and 2 <= df[c].nunique() <= 15 and "id" not in c.lower()]
if cat_cols:
    cat = cat_cols[0]
    opts = sorted(df[cat].unique())
    sel_cat = st.sidebar.multiselect(cat, opts, default=opts)
    mask_cat = df[cat].isin(sel_cat)
else:
    mask_cat = pd.Series(True, index=df.index)

f = feats[0]
lo, hi = float(df[f].min()), float(df[f].max())
rng = st.sidebar.slider(f, lo, hi, (lo, hi))

d = df[df.cluster_name.isin(sel_clusters) & mask_cat & df[f].between(*rng)]
page = st.sidebar.radio("Página", ["1. Análisis general", "2. Segmentación K-Means"])

st.title("🎧 Perfiles de oyentes y hábitos de escucha en streaming")

if d.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# ---------- KPIs (4) ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Usuarios", f"{len(d):,}")
k2.metric("Clústeres", d.cluster_name.nunique())
k3.metric(f"Promedio {feats[0]}", f"{d[feats[0]].mean():.2f}")
k4.metric(f"Promedio {feats[1]}", f"{d[feats[1]].mean():.2f}")

# ================= PÁGINA 1 =================
if page.startswith("1"):
    st.subheader("Análisis general del dataset")
    c1, c2 = st.columns(2)
    v = c1.selectbox("Variable a explorar", feats)
    c1.plotly_chart(px.histogram(d, x=v, nbins=30, title=f"Distribución de {v}"), use_container_width=True)
    c2.plotly_chart(px.imshow(d[feats].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                              title="Correlación entre variables"), use_container_width=True)
    if cat_cols:
        c3, c4 = st.columns(2)
        counts = d[cat_cols[0]].value_counts().reset_index()
        counts.columns = [cat_cols[0], "usuarios"]
        c3.plotly_chart(px.bar(counts, x=cat_cols[0], y="usuarios", title=f"Usuarios por {cat_cols[0]}"),
                        use_container_width=True)
        c4.plotly_chart(px.box(d, y=v, x=cat_cols[0], title=f"{v} por {cat_cols[0]}"),
                        use_container_width=True)
    st.subheader("Estadísticas descriptivas")
    st.dataframe(d[feats].describe().T.round(2), use_container_width=True)

# ================= PÁGINA 2 =================
else:
    st.subheader(f"Segmentación mediante K-Means (K = {K}, silhouette = {meta['silhouette']:.3f})")
    c1, c2 = st.columns(2)
    dist = d.cluster_name.value_counts().reset_index()
    dist.columns = ["cluster_name", "usuarios"]
    c1.plotly_chart(px.bar(dist, x="cluster_name", y="usuarios", color="cluster_name",
                           title="Distribución de usuarios por clúster"), use_container_width=True)
    c2.plotly_chart(px.scatter(d, x=feats[0], y=feats[1], color="cluster_name", opacity=.5,
                            render_mode="webgl", title="Agrupamientos de usuarios"),
                use_container_width=True)

    c3, c4 = st.columns(2)
    means = d.groupby("cluster_name")[feats].mean()
    norm = (means - means.min()) / (means.max() - means.min() + 1e-9)
    c3.plotly_chart(px.bar(norm.reset_index().melt("cluster_name"), x="variable", y="value",
                           color="cluster_name", barmode="group",
                           title="Perfil por clúster (promedios normalizados 0-1)"),
                    use_container_width=True)
    v2 = c4.selectbox("Variable para comparar", feats, key="v2")
    c4.plotly_chart(px.box(d, x="cluster_name", y=v2, color="cluster_name", title=f"{v2} por clúster"),
                    use_container_width=True)

    st.subheader("Estadísticas descriptivas por clúster")
    st.dataframe(d.groupby("cluster_name")[feats].agg(["mean", "median", "std"]).round(2),
                 use_container_width=True)

    st.subheader("Descripción de los grupos")
    DESC = {
        "Críticos": "Escuchan cerca del promedio (10.6 h/sem) y saltan poco, pero califican muy mal las recomendaciones (2.1 de 5). Son el grupo con mayor riesgo de abandono por insatisfacción.",
        "Skippers": "Los que más escuchan (11.7 h/sem) y más saltan (13.1 skips/día). Buscan constantemente algo que les guste y no lo encuentran a la primera.",
        "Playlists Creadors": "Crean muchas playlists (11.4, contra unas 7 en el resto) y escuchan 10.7 h/sem. Usuarios organizados y comprometidos con la plataforma.",
        "Casuales": "Escuchan menos (7.4 h/sem), saltan poco (8.2/día) y están satisfechos con las sugerencias (4.2 de 5). Es el grupo más grande (28.5%).",
    }
    for n in names:
        st.markdown(f"**{n}**: {DESC.get(n, '')}")

    st.subheader("Hallazgos y recomendaciones")
    st.markdown("""
**Hallazgos**
1. Los Skippers saltan 13.1 canciones al día, un 60% más que los Casuales (8.2), y además son quienes más escuchan.
2. El 22.4% de los usuarios (Críticos) califica las recomendaciones con 2.1/5 aunque consume música normalmente: la insatisfacción no depende de cuánto escuchan.
3. Los cuatro grupos tienen la misma inactividad (1.5 meses) y la misma edad (38 años): ninguna de las dos variables explica el perfil de escucha.
4. Los Playlists Creadors crean un 42% más de playlists que el promedio general (11.4 contra 8.0).

**Recomendaciones**
1. Críticos: pedir feedback (me gusta/no me gusta) y recalibrar sus recomendaciones con campañas tipo Discover Weekly personalizado.
2. Skippers: ofrecer recomendaciones basadas en estado de ánimo y previsualizaciones para reducir los skips.
3. Playlists Creadors: potenciar playlists colaborativas y compartidas, y la función más deseada por los usuarios (playlists automáticas por estado de ánimo).
4. Casuales: campañas de descubrimiento y notificaciones para aumentar su tiempo de escucha.
""")