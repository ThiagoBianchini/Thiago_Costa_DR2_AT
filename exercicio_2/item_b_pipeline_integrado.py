"""
Exercício 2 — Item B
Pipeline integrado sequencial: undistort → segmentação HSV → features ORB →
detector HOG/Haar → classificação DNN, com medição de tempo por etapa.
"""

import time
import cv2
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dnn_utils import (
    read_pipeline_frame,
    DEFAULT_K,
    DEFAULT_DIST,
    segment_red_hsv,
    orb_features,
    hog_or_haar_detector,
    predict_opencv,
    topk,
    draw_top3,
    load_labels,
    OUT,
    ensure_dirs
)

ensure_dirs()


# ============================================================
# ETAPA 1 — Captura do frame (sintético ou real)
# ============================================================

frame, origem = read_pipeline_frame()
tempos = {}
print(f"Origem do frame: {origem}\n")


# ============================================================
# ETAPA 2 — Correção de distorção (undistort)
# ============================================================

t0 = time.perf_counter()
frame_corrigido = cv2.undistort(frame, DEFAULT_K, DEFAULT_DIST)
tempos["undistort"] = (time.perf_counter() - t0) * 1000


# ============================================================
# ETAPA 3 — Segmentação de ROI por cor HSV
# ============================================================

t0 = time.perf_counter()
mask, roi_hsv = segment_red_hsv(frame_corrigido)
tempos["HSV"] = (time.perf_counter() - t0) * 1000

vis = frame_corrigido.copy()

if roi_hsv:
    x, y, w, h = roi_hsv
    roi_img = frame_corrigido[y:y + h, x:x + w]
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 3)
    cv2.putText(vis, "ROI HSV", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 120), 2)
else:
    roi_img = frame_corrigido
    x, y, w, h = 0, 0, frame_corrigido.shape[1], frame_corrigido.shape[0]


# ============================================================
# ETAPA 4 — Extração de features ORB
# ============================================================

t0 = time.perf_counter()
kp, des = orb_features(roi_img)
tempos["ORB"] = (time.perf_counter() - t0) * 1000

for p in kp[:60]:
    px, py = int(p.pt[0] + x), int(p.pt[1] + y)
    cv2.circle(vis, (px, py), 3, (0, 255, 0), -1)


# ============================================================
# ETAPA 5 — Detecção clássica HOG+SVM e Haar Cascade
# ============================================================

t0 = time.perf_counter()
boxes = hog_or_haar_detector(frame_corrigido)
tempos["HOG_Haar"] = (time.perf_counter() - t0) * 1000

for nome, bx, by, bw, bh in boxes:
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
    cv2.putText(vis, nome, (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)


# ============================================================
# ETAPA 6 — Classificação da ROI com OpenCV DNN
# ============================================================

t0 = time.perf_counter()
try:
    prob = predict_opencv(roi_img)
    tempos["DNN"] = (time.perf_counter() - t0) * 1000
    vis = draw_top3(vis, topk(prob, 3), load_labels())
except FileNotFoundError:
    tempos["DNN"] = 0.0
    cv2.putText(
        vis, "DNN: modelo nao encontrado", (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )
    print("AVISO: Modelo TFLite não encontrado. Execute item_a primeiro.")


# ============================================================
# ETAPA 7 — Anotação de tempos no frame
# ============================================================

y0 = vis.shape[0] - 140
cv2.rectangle(vis, (10, y0), (380, vis.shape[0] - 10), (255, 255, 255), -1)
cv2.rectangle(vis, (10, y0), (380, vis.shape[0] - 10), (0, 0, 0), 1)

for i, (nome, ms) in enumerate(tempos.items()):
    cv2.putText(
        vis, f"{nome}: {ms:.1f} ms",
        (22, y0 + 24 + i * 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2
    )

total_ms = sum(tempos.values())
cv2.putText(
    vis, f"Total: {total_ms:.1f} ms",
    (22, y0 + 24 + len(tempos) * 20),
    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 2
)


# ============================================================
# ETAPA 8 — Salvamento e impressão de resultados
# ============================================================

saida = OUT / "item_b_pipeline_integrado.jpg"
cv2.imwrite(str(saida), vis)

print("=" * 60)
print("PIPELINE INTEGRADO — Tempo de cada etapa")
print("=" * 60)
for nome, ms in tempos.items():
    print(f"  {nome:<12s}: {ms:7.2f} ms")
print(f"  {'Total':<12s}: {total_ms:7.2f} ms")
print("=" * 60)
print(f"Frame final salvo em: {saida}")
