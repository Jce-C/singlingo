"""
train_model.py — Entrena el clasificador para cada idioma.

v6: 96 features — 48 v6 mano derecha + 48 v6 mano izquierda.
    Features invariantes a rotación: ángulos articulares 3D + distancias normalizadas.

Clasificador: MLPClassifier (red neuronal) — mucho más preciso que LinearSVC.

Augmentación 3D:
  - Ruido en landmarks normalizados
  - Rotaciones aleatorias 3D del esqueleto de mano
  - Escala variable
  - Factor ×30 para datos reales capturados

Uso:
    python scripts/train_model.py          # entrena todos
    python scripts/train_model.py lsc      # solo LSC
"""
import csv
import json
import sys
from typing import Tuple
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TRAINING_DIR  = ROOT / "data" / "training"
COLLECTED_DIR = ROOT / "data" / "collected"
MODELS_DIR    = ROOT / "data" / "models"
MODELS_DIR.mkdir(exist_ok=True)

COUNTRIES = ["asl", "lsc", "bsl"]

FEATURE_SIZE_V5      = 78
FEATURE_SIZE_V5_TWO  = 156
FEATURE_SIZE_V6_ONE  = 48
FEATURE_SIZE_V6_TWO  = 96

AUGMENT_FACTOR = 30   # ×30 para datos reales capturados


def _rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    """Matriz de rotación 3D (ángulos en radianes)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _augment_landmarks_3d(
    pts_3d: np.ndarray,
    rng: np.random.RandomState,
    noise_std: float = 0.02,
    rot_deg: float = 35.0,
    scale_range: Tuple = (0.88, 1.12),
) -> np.ndarray:
    """
    Aumenta un array (21, 3) de landmarks normalizados con:
      - Rotación aleatoria 3D (simula la mano en cualquier orientación)
      - Ruido gaussiano
      - Escala variable
    """
    from typing import Tuple  # noqa

    aug = pts_3d.copy()

    # Rotación 3D aleatoria
    rx = rng.uniform(-np.radians(rot_deg), np.radians(rot_deg))
    ry = rng.uniform(-np.radians(rot_deg), np.radians(rot_deg))
    rz = rng.uniform(-np.radians(rot_deg), np.radians(rot_deg))
    R = _rotation_matrix_3d(rx, ry, rz)
    aug = (R @ aug.T).T

    # Escala
    scale = rng.uniform(scale_range[0], scale_range[1])
    aug *= scale

    # Ruido gaussiano por articulación
    aug += rng.normal(0, noise_std, aug.shape)

    # Re-normalizar (wrist al origen)
    aug -= aug[0].copy()
    palm = np.linalg.norm(aug[9])   # MIDDLE_MCP
    if palm > 1e-6:
        aug /= palm

    return aug


def augment_features_3d(
    X_raw_pts: np.ndarray,
    y: np.ndarray,
    factor: int = AUGMENT_FACTOR,
) -> tuple:
    """
    Aumenta X_raw_pts (N, 21*3) aplicando rotaciones 3D + ruido.
    Cada muestra se multiplica por `factor` variantes.
    """
    from core.landmarks import extract_features_v6, FEATURE_SIZE_V6_ONE

    N = len(X_raw_pts)
    rng = np.random.RandomState(42)

    all_feats = []
    all_labels = []

    for i in range(N):
        pts = X_raw_pts[i].reshape(21, 3)
        orig_feat = extract_features_v6(pts)
        all_feats.append(orig_feat)
        all_labels.append(y[i])

        for aug_i in range(factor - 1):
            noise = 0.010 + (aug_i % 7) * 0.006
            rot   = 25.0  + (aug_i % 5) * 5.0
            aug_pts = _augment_landmarks_3d(pts, rng, noise_std=noise, rot_deg=rot)
            aug_feat = extract_features_v6(aug_pts)
            all_feats.append(aug_feat)
            all_labels.append(y[i])

    return np.array(all_feats), np.array(all_labels)


def load_collected_real_data(country: str, label_map: dict):
    """
    Carga datos reales capturados desde la app y los aumenta con rotaciones 3D.
    Retorna (X_v6, y) con features v6 rotation-invariant.
    """
    from core.landmarks import normalize_landmarks

    country_dir = COLLECTED_DIR / country
    if not country_dir.exists():
        return None, None

    id_to_idx = {v: int(k) for k, v in label_map.items()}

    X_raw, y_raw = [], []
    for jsonl_file in country_dir.glob("*.jsonl"):
        sign_id = jsonl_file.stem
        if sign_id not in id_to_idx:
            continue
        idx = id_to_idx[sign_id]
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    lms   = entry.get("landmarks", [])
                    if len(lms) != 21:
                        continue
                    norm = normalize_landmarks(lms)
                    X_raw.append(norm.flatten())    # (63,) → store 3D pts
                    y_raw.append(idx)
                except Exception:
                    pass

    if not X_raw:
        return None, None

    X_raw = np.array(X_raw, dtype=np.float64)   # (N, 63)
    y_raw = np.array(y_raw)

    # Pad to (N, 63) — normalize_landmarks returns (21, 3)
    # already 63 from flatten()

    print(f"  Reales: {len(X_raw):,} muestras × {AUGMENT_FACTOR} aug = {len(X_raw) * AUGMENT_FACTOR:,}")
    X_aug, y_aug = augment_features_3d(X_raw, y_raw, factor=AUGMENT_FACTOR)

    # Pad a 96 features (mano izquierda = ceros) para compatibilidad con modelo two-hands
    if X_aug.shape[1] == FEATURE_SIZE_V6_ONE:
        X_aug = np.concatenate(
            [X_aug, np.zeros((len(X_aug), FEATURE_SIZE_V6_ONE), dtype=np.float64)], axis=1
        )

    return X_aug, y_aug


def load_training_data(country: str):
    """Carga datos sintéticos (v6) + datos reales aumentados."""
    synth_csv      = TRAINING_DIR / f"synthetic_{country}.csv"
    label_map_path = TRAINING_DIR / f"label_map_{country}.json"

    if not synth_csv.exists():
        raise FileNotFoundError(
            f"No se encontró {synth_csv}\n"
            f"Ejecuta primero: python scripts/generate_data.py"
        )
    if not label_map_path.exists():
        raise FileNotFoundError(f"No se encontró {label_map_path}")

    with open(label_map_path) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    def _read_csv_v6(path):
        """Lee CSV con features v6 (96 cols) o v5 y convierte a v6."""
        from core.landmarks import extract_features_v6

        X, y = [], []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_idx = int(row["label_idx"])
                feat_keys = [k for k in row.keys() if k not in ("label_idx", "sign_id")]
                n = len(feat_keys)
                if n < 42:
                    continue

                if n >= FEATURE_SIZE_V6_TWO:
                    features = [float(row[k]) for k in feat_keys[:FEATURE_SIZE_V6_TWO]]
                elif n >= FEATURE_SIZE_V6_ONE:
                    feats = [float(row[k]) for k in feat_keys[:FEATURE_SIZE_V6_ONE]]
                    features = feats + [0.0] * FEATURE_SIZE_V6_ONE
                elif n >= FEATURE_SIZE_V5_TWO:
                    # Datos sintéticos v5 viejos — tomar solo las primeras 63 como landmarks
                    # y re-extraer v6
                    raw_feats = [float(row[k]) for k in feat_keys[:FEATURE_SIZE_V5_TWO]]
                    features = raw_feats[:FEATURE_SIZE_V6_TWO] + [0.0] * max(0, FEATURE_SIZE_V6_TWO - len(raw_feats))
                else:
                    feats = [float(row[k]) for k in feat_keys[:n]]
                    features = feats + [0.0] * (FEATURE_SIZE_V6_TWO - len(feats))

                X.append(features[:FEATURE_SIZE_V6_TWO])
                y.append(label_idx)

        return np.array(X, dtype=np.float64), np.array(y)

    X_synth, y_synth = _read_csv_v6(synth_csv)
    print(f"  Sintéticos : {len(X_synth):,} muestras ({X_synth.shape[1]} features)")

    X_real, y_real = load_collected_real_data(country, label_map)

    if X_real is not None:
        real_classes = set(y_real.tolist())
        mask = np.array([yi not in real_classes for yi in y_synth])
        X_extra = X_synth[mask]
        y_extra = y_synth[mask]

        if len(X_extra) > 0:
            print(f"  Extra sinté.: {len(X_extra):,} (señas sin datos reales)")
            X = np.vstack([X_real, X_extra])
            y = np.concatenate([y_real, y_extra])
        else:
            X, y = X_real, y_real
        print(f"  TOTAL      : {len(X):,} muestras")
    else:
        X, y = X_synth, y_synth
        print(f"  TOTAL      : {len(X):,} muestras (solo sintéticos)")

    return X, y, label_map


def train_country(country: str):
    from sklearn.model_selection import train_test_split
    from core.classifier import SignClassifier

    print(f"\n  {'─'*52}")
    print(f"  Entrenando: {country.upper()}")
    print(f"  {'─'*52}")

    X, y, label_map = load_training_data(country)
    n_classes = len(np.unique(y))
    print(f"  Clases: {n_classes} señas | Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )

    clf = SignClassifier(country)
    clf.label_map = label_map

    print(f"  Entrenando MLPClassifier (256,128,64)...")
    train_acc = clf.train(X_train, y_train, use_mlp=True)

    test_preds = clf.pipeline.predict(X_test)
    test_acc   = float(np.mean(test_preds == y_test))

    print(f"  Train acc  : {train_acc:.1%}")
    print(f"  Test acc   : {test_acc:.1%}")

    quality = "🟢" if test_acc >= 0.70 else "🟡" if test_acc >= 0.55 else "🔴"
    if test_acc < 0.55:
        print(f"  {quality} Accuracy baja — captura datos reales desde la app.")
    elif test_acc < 0.70:
        print(f"  {quality} Accuracy media — más datos reales mejoran el modelo.")
    else:
        print(f"  {quality} Accuracy buena.")

    clf.save()
    print(f"  ✅ Modelo guardado: data/models/classifier_{country}.pkl")
    return test_acc


def main():
    target    = sys.argv[1] if len(sys.argv) > 1 else "all"
    countries = COUNTRIES if target == "all" else [target.lower()]

    print("=" * 65)
    print("  SignLingo — Entrenamiento v6 (rotation-invariant)")
    print("  96 features: 48 mano derecha + 48 mano izquierda")
    print("  MLPClassifier (256,128,64) | Augmentación 3D ×30")
    print("=" * 65)

    results = {}
    for country in countries:
        if country not in COUNTRIES:
            print(f"\n  ❌ País desconocido: {country}")
            continue
        try:
            acc = train_country(country)
            results[country] = acc
        except FileNotFoundError as e:
            print(f"\n  ❌ {e}")
            results[country] = None

    print("\n" + "=" * 65)
    print("  Resumen:")
    for c, acc in results.items():
        if acc is not None:
            bar     = "█" * int(acc * 20)
            quality = "🟢" if acc >= 0.70 else "🟡" if acc >= 0.55 else "🔴"
            print(f"  {c.upper():5s} {quality} {bar:<20s} {acc:.1%}")
        else:
            print(f"  {c.upper():5s} ✗ No entrenado")
    print("=" * 65)


if __name__ == "__main__":
    main()
