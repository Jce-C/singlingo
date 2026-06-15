"""
generate_data.py — Genera datos de entrenamiento sintéticos con features v6.

v6: 96 features (48 mano derecha + 48 mano izquierda).
Features rotation-invariant: ángulos articulares 3D + distancias normalizadas.

Mejoras vs v5:
  - Genera poses 3D (con coordenada z simulada)
  - Aplica rotaciones 3D aleatorias durante la generación
  - Features v6 invariantes a la dirección de la mano
  - Más muestras por seña (900)

Uso:
    python scripts/generate_data.py
    python scripts/generate_data.py lsc
"""
import json
import csv
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SIGNS_DIR    = ROOT / "data" / "signs"
TRAINING_DIR = ROOT / "data" / "training"
TRAINING_DIR.mkdir(exist_ok=True)

N_SAMPLES_PER_SIGN = 900
RANDOM_SEED = 42
COUNTRIES = ["asl", "lsc", "bsl"]

# ── Plantilla 3D de una mano neutral ────────────────────────────────────────
# Coordenadas (x, y, z) normalizadas con wrist en origen, palm_size=1
_WRIST_3D      = np.array([ 0.000,  0.000,  0.000])
_THUMB_CMC_3D  = np.array([-0.210, -0.370,  0.020])
_THUMB_MCP_3D  = np.array([-0.315, -0.580,  0.025])
_INDEX_MCP_3D  = np.array([-0.145, -0.940, -0.015])
_MIDDLE_MCP_3D = np.array([ 0.000, -1.000, -0.020])
_RING_MCP_3D   = np.array([ 0.138, -0.940, -0.015])
_PINKY_MCP_3D  = np.array([ 0.245, -0.855, -0.010])

_SEG_LEN = {
    "thumb":  [0.195, 0.175, 0.155],
    "index":  [0.360, 0.240, 0.205],
    "middle": [0.380, 0.255, 0.215],
    "ring":   [0.355, 0.240, 0.200],
    "pinky":  [0.290, 0.190, 0.155],
}

# Dirección de cada dedo cuando está extendido (ángulo en plano XY)
_DIR_UP_DEG = {
    "thumb":  105.0,
    "index":  177.0,
    "middle": 180.0,
    "ring":   184.0,
    "pinky":  188.0,
}
_DIR_FOLD_DEG = {
    "thumb":  36.0,
    "index":  108.0,
    "middle": 108.0,
    "ring":   108.0,
    "pinky":  108.0,
}


def _rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    """Matriz de rotación 3D compuesta (ángulos en radianes)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _seg_positions_3d(base: np.ndarray, seg_lens: list, angle_deg: float, z_slope: float) -> list:
    """Genera posiciones 3D de los segmentos de un dedo."""
    angle_rad = np.radians(angle_deg)
    d_xy = np.array([np.sin(angle_rad), -np.cos(angle_rad)])
    positions = []
    cur = base.copy()
    for i, seg in enumerate(seg_lens):
        cur = cur + np.array([d_xy[0] * seg, d_xy[1] * seg, z_slope * seg * (i + 1)])
        positions.append(cur.copy())
    return positions


def generate_hand_pose_3d(
    feature_template: dict,
    finger_states: list,
    noise_std: float = 0.03,
    rotation_3d: np.ndarray = None,
    scale: float = 1.0,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """
    Genera una pose 3D de mano (21, 3) a partir de un template de features.

    Args:
        feature_template: dict con ext_ratios y configuración de la seña
        finger_states:    [thumb, index, middle, ring, pinky] extendidos
        noise_std:        desviación del ruido gaussiano
        rotation_3d:      matriz 3×3 de rotación (si None, sin rotación)
        scale:            factor de escala
        rng:              generador aleatorio

    Returns:
        pts: array (21, 3) normalizado
    """
    if rng is None:
        rng = np.random.RandomState()

    ft = feature_template
    thumb_up, index_up, middle_up, ring_up, pinky_up = finger_states

    pts = np.zeros((21, 3))
    pts[0]  = _WRIST_3D.copy()
    pts[1]  = _THUMB_CMC_3D.copy()
    pts[2]  = _THUMB_MCP_3D.copy()
    pts[5]  = _INDEX_MCP_3D.copy()
    pts[9]  = _MIDDLE_MCP_3D.copy()
    pts[13] = _RING_MCP_3D.copy()
    pts[17] = _PINKY_MCP_3D.copy()

    # ── Pulgar ────────────────────────────────────────────────────────────
    ext_t = ft.get("thumb_ext", 1.2)
    horiz = ft.get("thumb_horiz", 0.5)

    if thumb_up or ext_t > 1.1:
        base_angle = 35 + horiz * 55
        thumb_angle = base_angle + rng.uniform(-8, 8)
        z_slope_t = -0.04
    else:
        if horiz > 0.2:
            base_angle = 20 + horiz * 50 + rng.uniform(-8, 8)
        else:
            base_angle = 15 + rng.uniform(-6, 6)
        thumb_angle = base_angle
        z_slope_t = 0.08  # pulgar doblado se aleja de la cámara

    thumb_len   = _SEG_LEN["thumb"]
    thumb_scale = ext_t / 1.5
    thumb_segs  = [s * np.clip(thumb_scale, 0.4, 1.4) for s in thumb_len]
    base_t      = pts[2].copy()
    thumb_joints = _seg_positions_3d(base_t, thumb_segs, thumb_angle, z_slope_t)
    for i, j in enumerate(thumb_joints):
        pts[3 + i] = j

    pts[3, :2] += rng.normal(0, 0.04, 2)

    # ── Dedos ─────────────────────────────────────────────────────────────
    finger_defs = [
        ("index",  [5, 6, 7, 8],    index_up,  ft.get("index_ext",  1.5), ft.get("spread", 0.5)),
        ("middle", [9, 10, 11, 12], middle_up, ft.get("middle_ext", 1.5), 0.0),
        ("ring",   [13, 14, 15, 16],ring_up,   ft.get("ring_ext",   1.5), 0.0),
        ("pinky",  [17, 18, 19, 20],pinky_up,  ft.get("pinky_ext",  1.5), 0.0),
    ]

    for fname, indices, is_up, ext_ratio, lateral in finger_defs:
        base_idx = indices[0]
        segs = _SEG_LEN[fname]

        if is_up or ext_ratio > 1.3:
            angle = _DIR_UP_DEG[fname] + rng.uniform(-7, 7)
            if fname == "index":
                angle += lateral * 6
            z_slope = -0.03 + rng.uniform(-0.01, 0.01)
        else:
            extra_fold = (1.3 - ext_ratio) * 15
            angle   = _DIR_FOLD_DEG[fname] + extra_fold + rng.uniform(-5, 5)
            z_slope = 0.06 + (1.3 - ext_ratio) * 0.04

        seg_scale   = np.clip(ext_ratio / 1.8, 0.6, 1.2)
        scaled_segs = [s * seg_scale for s in segs]

        base   = pts[base_idx].copy()
        joints = _seg_positions_3d(base, scaled_segs, angle, z_slope)
        for i, idx in enumerate(indices[1:]):
            pts[idx] = joints[i]

    # ── Escala ────────────────────────────────────────────────────────────
    if abs(scale - 1.0) > 1e-6:
        pts *= scale

    # ── Rotación 3D ───────────────────────────────────────────────────────
    if rotation_3d is not None:
        pts = (rotation_3d @ pts.T).T

    # ── Ruido gaussiano ───────────────────────────────────────────────────
    noise = rng.normal(0, noise_std, size=pts.shape)
    tip_mask = np.zeros(21)
    tip_mask[[4, 8, 12, 16, 20]] = 1.5
    noise *= (1.0 + tip_mask[:, None])
    pts += noise

    # ── Re-normalizar (wrist al origen, palm_size=1) ──────────────────────
    pts -= pts[0].copy()
    palm = np.linalg.norm(pts[9])    # MIDDLE_MCP
    if palm > 1e-6:
        pts /= palm

    return pts


def pose_to_features_v6(pts_3d: np.ndarray) -> np.ndarray:
    """Convierte array (21, 3) en vector de 48 features v6."""
    from core.landmarks import extract_features_v6
    return extract_features_v6(pts_3d)


def pose_to_features_v6_two(pts_right: np.ndarray, pts_left: np.ndarray) -> np.ndarray:
    """Convierte dos poses (21,3) en 96 features v6-two."""
    from core.landmarks import extract_features_v6
    return np.concatenate([extract_features_v6(pts_right), extract_features_v6(pts_left)])


def generate_country_data(country: str):
    path = SIGNS_DIR / f"{country}.json"
    if not path.exists():
        print(f"  [SKIP] {country}.json no encontrado")
        return

    with open(path) as f:
        signs = json.load(f)["signs"]

    rng = np.random.RandomState(RANDOM_SEED)
    rows = []
    label_map = {}

    for label_idx, sign in enumerate(signs):
        finger_states = sign.get("finger_states")
        if not finger_states or len(finger_states) < 5:
            print(f"  [WARN] Sin finger_states para {sign['id']}, saltando")
            continue

        ft = sign.get("feature_template", {
            "thumb_ext": 1.0, "index_ext": 1.5, "middle_ext": 1.5,
            "ring_ext": 1.5, "pinky_ext": 1.5,
            "pinch_dist": 0.8, "spread": 0.5, "thumb_horiz": 0.4,
        })

        two_handed          = sign.get("two_handed", False)
        finger_states_other = sign.get("finger_states_other", [False] * 5)
        ft_other            = sign.get("feature_template_other", ft)

        label_map[label_idx] = sign["id"]

        for i in range(N_SAMPLES_PER_SIGN):
            # Rotación 3D aleatoria — clave para rotation-invariance
            # Ángulos más grandes hacia el final del batch para mayor cobertura
            frac = i / N_SAMPLES_PER_SIGN
            rot_range  = np.radians(30 + frac * 20)   # 30°→50° según iteración
            noise_base = 0.030 + 0.020 * frac
            scale_var  = 0.15  + 0.05  * frac

            rx = rng.uniform(-rot_range, rot_range)
            ry = rng.uniform(-rot_range, rot_range)
            rz = rng.uniform(-np.radians(45), np.radians(45))
            R  = _rotation_matrix_3d(rx, ry, rz)

            scale    = rng.uniform(1.0 - scale_var, 1.0 + scale_var)
            ft_varied = {k: v * rng.uniform(0.90, 1.10) for k, v in ft.items()}

            pts_right = generate_hand_pose_3d(
                ft_varied, finger_states,
                noise_std=noise_base, rotation_3d=R, scale=scale, rng=rng,
            )

            if two_handed and finger_states_other:
                rx_l = rng.uniform(-np.radians(25), np.radians(25))
                ry_l = rng.uniform(-np.radians(25), np.radians(25))
                rz_l = rng.uniform(-np.radians(35), np.radians(35))
                R_l  = _rotation_matrix_3d(rx_l, ry_l, rz_l)
                ft_oth_varied = {k: v * rng.uniform(0.90, 1.10) for k, v in ft_other.items()}
                pts_left = generate_hand_pose_3d(
                    ft_oth_varied, finger_states_other,
                    noise_std=noise_base, rotation_3d=R_l,
                    scale=rng.uniform(0.90, 1.10), rng=rng,
                )
                feat = pose_to_features_v6_two(pts_right, pts_left)
            else:
                feat_right = pose_to_features_v6(pts_right)
                feat_left  = np.zeros(48, dtype=np.float64)
                feat = np.concatenate([feat_right, feat_left])

            rows.append([label_idx, sign["id"]] + feat.tolist())

    feature_names = [f"f{i}" for i in range(96)]
    headers = ["label_idx", "sign_id"] + feature_names

    out_csv = TRAINING_DIR / f"synthetic_{country}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    label_map_path = TRAINING_DIR / f"label_map_{country}.json"
    with open(label_map_path, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)

    two_hand_count = sum(1 for s in signs if s.get("two_handed", False))
    print(f"  ✓ {country.upper()}: {len(signs)} señas ({two_hand_count} bimanuales) "
          f"× {N_SAMPLES_PER_SIGN} = {len(rows)} muestras (96 features v6) → {out_csv.name}")


def main():
    target    = sys.argv[1] if len(sys.argv) > 1 else "all"
    countries = COUNTRIES if target == "all" else [target.lower()]

    print("=" * 65)
    print("  SignLingo — Generador de datos sintéticos (features v6)")
    print("  96 features (48×2): ángulos articulares 3D rotation-invariant")
    print(f"  {N_SAMPLES_PER_SIGN} muestras × rotaciones 3D aleatorias por seña")
    print("=" * 65)

    for country in countries:
        if country not in COUNTRIES:
            print(f"\n  ❌ País desconocido: {country}")
            continue
        print(f"\nGenerando datos para: {country.upper()}")
        generate_country_data(country)

    print(f"\n✅ Datos generados en {TRAINING_DIR}/")
    print("   Siguiente: python scripts/train_model.py")


if __name__ == "__main__":
    main()
