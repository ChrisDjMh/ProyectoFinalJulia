from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
IMG_DIR = ROOT / "images"
PROCESSED = PROCESSED_DIR / "spotify_clusters.csv"
FEATURES_JSON = PROCESSED_DIR / "features.json"

FEATURES = [
    "avg_listening_hours_per_week",
    "avg_skips_per_day",
    "playlists_created",
    "music_suggestion_rating_1_to_5",
]

# Variables de perfil: NO entran al modelo, solo describen los clústeres
EXTRA = ["months_inactive", "age"]

K_FINAL = 4
K_ALT = 3

# Llénalo después de ver los centroides. Ej:
# {0: "Oyentes intensos", 1: "Zappeadores", 2: "Curadores de playlists", 3: "Usuarios inactivos"}
CLUSTER_NAMES = {
    0: "Críticos",
    1: "Skippers",
    2: "Playlists Creadors",
    3: "Casuales",
}

RANDOM_STATE = 42


def pick_features(df):
    return FEATURES