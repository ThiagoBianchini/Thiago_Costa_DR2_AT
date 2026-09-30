"""
Módulo utilitário para geração de imagens sintéticas de calibração.

Cria um laboratório virtual de calibração com câmera simulada,
permitindo execução reprodutível sem dependência de hardware.
"""

from pathlib import Path
import cv2
import numpy as np

# ============================================================
# CONSTANTES DO PADRÃO DE CALIBRAÇÃO
# ============================================================

PATTERN_SIZE = (7, 6)
SQUARE_SIZE_M = 0.03
IMAGE_SIZE = (1280, 720)

TRUE_K = np.array([
    [900.0,   0.0, 640.0],
    [  0.0, 910.0, 360.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

TRUE_DIST = np.zeros((5, 1), dtype=np.float64)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def ensure_dirs(root):
    root = Path(root)
    (root / "dados_sinteticos").mkdir(exist_ok=True, parents=True)
    (root / "saidas").mkdir(exist_ok=True, parents=True)


def make_checkerboard(square_px=100):
    cols = PATTERN_SIZE[0] + 1
    rows = PATTERN_SIZE[1] + 1
    img = np.full((rows * square_px, cols * square_px), 255, np.uint8)
    for y in range(rows):
        for x in range(cols):
            if (x + y) % 2 == 0:
                cv2.rectangle(
                    img,
                    (x * square_px, y * square_px),
                    ((x + 1) * square_px, (y + 1) * square_px),
                    0, -1
                )
    return img


def euler_to_rvec(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    R = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    return rvec


def generate_view(
    rx=0, ry=0, rz=0,
    tx=0.0, ty=0.0, tz=0.65,
    image_size=IMAGE_SIZE,
    K=TRUE_K,
    background=220,
    blur=0.0,
    noise_std=0.0
):
    w, h = image_size
    texture = make_checkerboard(120)

    board_w = (PATTERN_SIZE[0] + 1) * SQUARE_SIZE_M
    board_h = (PATTERN_SIZE[1] + 1) * SQUARE_SIZE_M

    outer_3d = np.array([
        [0.0,     0.0,     0.0],
        [board_w, 0.0,     0.0],
        [board_w, board_h, 0.0],
        [0.0,     board_h, 0.0]
    ], dtype=np.float32)

    rvec = euler_to_rvec(rx, ry, rz)
    tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

    dst, _ = cv2.projectPoints(outer_3d, rvec, tvec, K, TRUE_DIST)
    dst = dst.reshape(-1, 2).astype(np.float32)

    src = np.array([
        [0, 0],
        [texture.shape[1] - 1, 0],
        [texture.shape[1] - 1, texture.shape[0] - 1],
        [0, texture.shape[0] - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)

    bg = np.full((h, w), background, dtype=np.uint8)
    warped = cv2.warpPerspective(texture, H, (w, h), borderValue=background)

    mask_src = np.full(texture.shape, 255, dtype=np.uint8)
    mask = cv2.warpPerspective(mask_src, H, (w, h), borderValue=0)
    bg[mask > 0] = warped[mask > 0]

    if blur > 0:
        k = int(max(3, round(blur) * 2 + 1))
        if k % 2 == 0:
            k += 1
        bg = cv2.GaussianBlur(bg, (k, k), blur)

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, bg.shape)
        bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR), rvec, tvec


# ============================================================
# GERAÇÃO DE DATASET E CALIBRAÇÃO
# ============================================================

def generate_dataset(root, n=18, noisy=False):
    root = Path(root)
    ensure_dirs(root)
    out = root / "dados_sinteticos"

    poses = [
        (  0,   0,   0, -0.12, -0.09, 0.70),
        ( -8,  10,   4, -0.10, -0.07, 0.65),
        ( 10, -12,  -5, -0.08, -0.06, 0.72),
        ( 15,   8,   8, -0.14, -0.08, 0.78),
        (-14, -10, -10, -0.07, -0.10, 0.68),
        ( 20,   5,  12, -0.11, -0.05, 0.82),
        (-18,  15,  -8, -0.09, -0.08, 0.75),
        (  5,  20,  15, -0.13, -0.07, 0.85),
        ( -5, -20, -15, -0.06, -0.06, 0.73),
        ( 12,  18,  -4, -0.10, -0.10, 0.90),
        (-12,  -8,  14, -0.08, -0.04, 0.80),
        ( 22, -15,   6, -0.15, -0.06, 0.88),
        (-20,  12,  10, -0.05, -0.09, 0.77),
        (  8, -22,   3, -0.12, -0.11, 0.84),
        ( -7,  25,  -6, -0.07, -0.05, 0.92),
        ( 16,  -5, -12, -0.11, -0.09, 0.74),
        (-16,   6,  11, -0.09, -0.03, 0.86),
        (  3, -10,  18, -0.13, -0.08, 0.79),
    ][:n]

    paths = []
    for i, p in enumerate(poses, start=1):
        image, _, _ = generate_view(
            *p,
            blur=0.8 if noisy else 0.0,
            noise_std=2.0 if noisy else 0.0
        )
        path = out / f"calib_{i:02d}.png"
        cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


def object_points():
    obj = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    obj[:, :2] = np.mgrid[
        0:PATTERN_SIZE[0],
        0:PATTERN_SIZE[1]
    ].T.reshape(-1, 2)
    obj *= SQUARE_SIZE_M
    return obj


def find_refined_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)
    if not ok:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def calibrate_from_paths(paths):
    objpoints = []
    imgpoints = []
    image_size = None
    used = []

    obj = object_points()

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        image_size = (img.shape[1], img.shape[0])
        ok, corners = find_refined_corners(img)
        if ok:
            objpoints.append(obj.copy())
            imgpoints.append(corners)
            used.append(path)

    if len(objpoints) < 3:
        raise RuntimeError("Poucas imagens válidas para calibração.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    return rms, K, dist, rvecs, tvecs, objpoints, imgpoints, used, image_size


def reprojection_errors(K, dist, rvecs, tvecs, objpoints, imgpoints):
    errors = []
    for obj, img, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = np.mean(np.linalg.norm(img.reshape(-1, 2) - projected.reshape(-1, 2), axis=1))
        errors.append(float(err))
    return errors

