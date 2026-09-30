"""
Exercício 1 — Item B
Realidade aumentada: detecção de tabuleiro, estimativa de pose via solvePnP
e sobreposição de cubo 3D virtual com faces coloridas em vídeo sintético.
"""

from pathlib import Path
import cv2
import numpy as np
from synthetic_utils import (
    generate_dataset,
    calibrate_from_paths,
    generate_view,
    find_refined_corners,
    object_points,
    SQUARE_SIZE_M,
    IMAGE_SIZE
)

ROOT = Path(__file__).resolve().parent


# ============================================================
# ETAPA 1 — Calibração prévia da câmera
# ============================================================

paths = generate_dataset(ROOT, n=18)
_, K, dist, *_ = calibrate_from_paths(paths)
print("Calibração concluída.")
print(f"K:\n{K}")
print(f"Distorção: {dist.ravel()[:5]}\n")


# ============================================================
# ETAPA 2 — Definição do cubo 3D (aresta = 1 quadrado)
# ============================================================

s = SQUARE_SIZE_M

cube_vertices = np.float32([
    [0, 0,  0], [s, 0,  0], [s, s,  0], [0, s,  0],
    [0, 0, -s], [s, 0, -s], [s, s, -s], [0, s, -s]
])

cube_faces = [
    ([0, 1, 2, 3], (0, 255, 0)),
    ([4, 5, 6, 7], (255, 0, 0)),
    ([0, 1, 5, 4], (0, 255, 255)),
    ([1, 2, 6, 5], (0, 0, 255)),
    ([2, 3, 7, 6], (255, 255, 0)),
    ([3, 0, 4, 7], (255, 0, 255)),
]

cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]


# ============================================================
# ETAPA 3 — Geração do vídeo AR com cubo sobreposto
# ============================================================

saida_video = ROOT / "saidas" / "item_b_ar_cubo_3d.avi"
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(str(saida_video), fourcc, 15.0, IMAGE_SIZE)

n_frames = 90
obj_pts = object_points()

print("Gerando vídeo de realidade aumentada...")
print("-" * 60)

for i in range(n_frames):
    a = i / (n_frames - 1)

    frame, _, _ = generate_view(
        rx=8 + 8 * np.sin(a * np.pi * 2),
        ry=-12 + 12 * np.sin(a * np.pi),
        rz=6 * np.sin(a * np.pi * 2),
        tx=-0.10 + 0.02 * np.sin(a * np.pi * 2),
        ty=-0.07,
        tz=0.75 + 0.05 * np.sin(a * np.pi * 2)
    )

    ok, corners = find_refined_corners(frame)

    if not ok:
        writer.write(frame)
        continue

    # --- Estimativa de pose com solvePnP ---
    _, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, dist)

    # --- Projeção dos vértices do cubo ---
    pts_2d, _ = cv2.projectPoints(cube_vertices, rvec, tvec, K, dist)
    p = pts_2d.reshape(-1, 2).astype(int)

    # --- Desenho das faces semitransparentes ---
    overlay = frame.copy()
    for vertex_ids, color in cube_faces:
        poly = np.array([p[v] for v in vertex_ids], np.int32)
        cv2.fillConvexPoly(overlay, poly, color)
    frame = cv2.addWeighted(overlay, 0.30, frame, 0.70, 0)

    # --- Desenho das arestas ---
    for a_idx, b_idx in cube_edges:
        cv2.line(frame, tuple(p[a_idx]), tuple(p[b_idx]), (30, 30, 30), 2)

    # --- Impressão de pose no terminal ---
    rvec_str = np.array2string(rvec.ravel(), precision=4, separator=", ")
    tvec_str = np.array2string(tvec.ravel(), precision=4, separator=", ")
    print(f"Frame {i+1:03d}  |  rvec: {rvec_str}  |  tvec [m]: {tvec_str}")

    writer.write(frame)

writer.release()
print("-" * 60)
print(f"Vídeo AR salvo em: {saida_video}")


# ============================================================
# ETAPA 4 — Imagem estática de referência com cubo
# ============================================================

frame_ref, _, _ = generate_view(14, -17, 8, -0.10, -0.07, 0.80)
ok, corners_ref = find_refined_corners(frame_ref)

if ok:
    _, rvec_ref, tvec_ref = cv2.solvePnP(obj_pts, corners_ref, K, dist)
    pts_ref, _ = cv2.projectPoints(cube_vertices, rvec_ref, tvec_ref, K, dist)
    p_ref = pts_ref.reshape(-1, 2).astype(int)

    overlay_ref = frame_ref.copy()
    for vertex_ids, color in cube_faces:
        poly = np.array([p_ref[v] for v in vertex_ids], np.int32)
        cv2.fillConvexPoly(overlay_ref, poly, color)
    frame_ref = cv2.addWeighted(overlay_ref, 0.30, frame_ref, 0.70, 0)

    for a_idx, b_idx in cube_edges:
        cv2.line(frame_ref, tuple(p_ref[a_idx]), tuple(p_ref[b_idx]), (30, 30, 30), 2)

    saida_img = ROOT / "saidas" / "item_b_cubo_referencia.png"
    cv2.imwrite(str(saida_img), frame_ref)
    print(f"Imagem de referência salva em: {saida_img}")
