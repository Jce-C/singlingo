"""
augment_data.py — Aumentación de datos reales de landmarks con features v4.

Convierte N muestras reales en ~30×N muestras aumentadas usando
transformaciones geométricas sobre los landmarks.

Uso:
    python scripts/augment_data.py lsc
    python scripts/augment_data.py all
"""
import json
import csv
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

COLLECTED_DIR = ROOT / "data" / "collected"
TRAINING_DIR  = ROOT / "data" / "training"
TRAINING_DIR.mkdir(exist_ok=True)

ROTATIONS  = [-0.26, -0.13, 0.0, 0.13, 0.26]
SCALES     = [0.88, 1.00, 1.12]
NOISE_STDS = [0.010, 0.020]
COUNTRIES  = ["asl", "lsc", "bsl"]

N_AUGMENTATIONS = len(ROTATIONS) * len(SCALES) * len(NOISE_STDS)  # 30


def augment_sample(pts: np.ndarray, rng: np.random.RandomState) -> list:
    """Genera 30 variantes de una muestra (21, 2) normalizada."""
    augmented = []
    for rot in ROTATIONS:
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        rotated = pts @ R.T
        for scale in SCALES:
            scaled = rotated * scale
            for noise_std in NOISE_STDS:
                noisy = scaled + rng.normal(0, noise_std, size=scaled.shape)
                augmented.append(noisy)
    return augmented


def normalize_raw(raw: list) -> np.ndarray:
    """Normaliza landmarks crudos a (21, 2)."""
    pts = np.array([[p["x"], p["y"]] for p in raw], dtype=np.float64)
    pts -= pts[0].copy()  # centrar en muñeca
    palm_size = np.linalg.norm(pts[9])
    if palm_size < 1e-6:
        palm_size = 1.0
    pts /= palm_size
    return pts


def pts_to_features_v4(pts: np.ndarray) -> list:
    """Convierte (21, 2) a vector de 63 features v4 (incluyendo geométricas)."""
    raw = [{"x": float(pts[i, 0]), "y": float(pts[i, 1]), "z": 0.0} for i in range(21)]
    from core.landmarks import normalize_landmarks, extract_features_v4
    norm = normalize_landmarks(raw)
    return extract_features_v4(norm).tolist()


def augment_country(country: str):
    collected_path = COLLECTED_DIR / country
    if not collected_path.exists():
        print(f"  [SKIP] No hay datos reales para {country.upper()}")
        return 0

    sign_files = list(collected_path.glob("*.jsonl"))
    if not sign_files:
        print(f"  [SKIP] {country.upper()}: no se encontraron archivos .jsonl")
        return 0

    label_map_path = TRAINING_DIR / f"label_map_{country}.json"
    if not label_map_path.exists():
        print(f"  [WARN] No hay label_map para {country}. Ejecuta generate_data.py primero.")
        return 0

    with open(label_map_path) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    reverse_map = {v: k for k, v in label_map.items()}

    rng = np.random.RandomState(42)
    rows = []
    total_real = 0
    total_aug  = 0

    for sign_file in sorted(sign_files):
        sign_id = sign_file.stem
        if sign_id not in reverse_map:
            print(f"  [WARN] '{sign_id}' no está en label_map, saltando")
            continue

        label_idx = reverse_map[sign_id]
        raw_samples = []

        with open(sign_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    lms = entry.get("landmarks", [])
                    if len(lms) == 21:
                        raw_samples.append(lms)
                except Exception:
                    pass

        if not raw_samples:
            continue

        total_real += len(raw_samples)
        n_aug = 0

        for raw in raw_samples:
            pts = normalize_raw(raw)
            for aug_pts in augment_sample(pts, rng):
                # v4: 63 features de mano derecha + 63 ceros de mano izquierda
                feat_right = pts_to_features_v4(aug_pts)
                feat_left  = [0.0] * 63
                rows.append([label_idx, sign_id] + feat_right + feat_left)
                n_aug += 1

        total_aug += n_aug
        print(f"    {sign_id}: {len(raw_samples)} reales × {N_AUGMENTATIONS} = {n_aug} aumentados")

    if not rows:
        print(f"  [SKIP] No se generaron datos para {country.upper()}")
        return 0

    feature_names = [f"f{i}" for i in range(126)]
    headers = ["label_idx", "sign_id"] + feature_names
    out_csv = TRAINING_DIR / f"augmented_{country}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"\n  ✅ {country.upper()}: {total_real} reales → {total_aug} aumentadas → {out_csv.name}")
    return total_aug


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    countries = COUNTRIES if target == "all" else [target.lower()]

    print("=" * 60)
    print("  SignLingo — Aumentación de datos reales (features v4)")
    print(f"  ×{N_AUGMENTATIONS} por muestra (rot × escala × ruido)")
    print("=" * 60)

    total = 0
    for country in countries:
        if country not in COUNTRIES:
            print(f"  [ERROR] País desconocido: {country}")
            continue
        print(f"\nAumentando {country.upper()}...")
        total += augment_country(country)

    if total > 0:
        print(f"\n✅ Total: {total:,} muestras aumentadas")
        print("   Siguiente: python scripts/train_model.py")
    else:
        print("\n⚠️  No hay datos reales todavía. Graba tus señas desde la app.")


if __name__ == "__main__":
    main()
