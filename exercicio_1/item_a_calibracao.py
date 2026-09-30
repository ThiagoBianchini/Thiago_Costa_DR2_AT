"""
Exercício 1 — Item A
Calibração de câmera: obtenção da matriz intrínseca, coeficientes de distorção,
erro de reprojeção e painel comparativo original vs. corrigida.
"""

from pathlib import Path
import cv2
import numpy as np
from synthetic_utils import (
    generate_dataset,
    calibrate_from_paths,
    reprojection_errors,
    PATTERN_SIZE,
    SQUARE_SIZE_M,
    TRUE_K
)

ROOT = Path(__file__).resolve().parent


# ============================================================
# ETAPA 1 — Geração de 18 imagens sintéticas de calibração
# ============================================================

paths = generate_dataset(ROOT, n=18)
print(f"Imagens geradas: {len(paths)}")


# ============================================================
# ETAPA 2 — Calibração da câmera
# ============================================================

rms, K, dist, rvecs, tvecs, objpoints, imgpoints, used, image_size = \
    calibrate_from_paths(paths)


# ============================================================
# ETAPA 3 — Erro de reprojeção por imagem e erro médio
# ============================================================

errors = reprojection_errors(K, dist, rvecs, tvecs, objpoints, imgpoints)

print("\n" + "=" * 60)
print("RESULTADOS DA CALIBRAÇÃO")
print("=" * 60)

# --- Matriz intrínseca K ---
# K[0,0] = fx (distância focal em pixels no eixo X)
# K[1,1] = fy (distância focal em pixels no eixo Y)
# K[0,2] = cx (coordenada X do ponto principal — centro óptico)
# K[1,2] = cy (coordenada Y do ponto principal — centro óptico)
# Valores ideais de cx e cy coincidem com o centro da imagem.
print("\nMatriz Intrínseca K:")
print(K)

# --- Coeficientes de distorção ---
# [k1, k2, p1, p2, k3]
# k1, k2, k3 = distorção radial (barrel/pincushion)
# p1, p2     = distorção tangencial (desalinhamento das lentes)
# Em câmeras de boa qualidade, todos devem ser próximos de zero.
coefs = dist.ravel()[:5]
print(f"\nCoeficientes de distorção [k1, k2, p1, p2, k3]:\n{coefs}")

# --- Erro de reprojeção ---
# Erro médio < 0.5 px é considerado excelente para robótica.
# Erro entre 0.5 e 1.0 px é aceitável para a maioria das aplicações.
# Erro > 1.0 px indica problemas na captura ou no padrão de calibração.
erro_medio = np.mean(errors)
print(f"\nErro de reprojeção por imagem (px):")
for i, err in enumerate(errors):
    print(f"  Imagem {i+1:02d}: {err:.4f}")
print(f"\nErro médio de reprojeção: {erro_medio:.4f} px")

if erro_medio < 0.5:
    print("Avaliação: EXCELENTE — adequado para odometria visual e SLAM.")
elif erro_medio < 1.0:
    print("Avaliação: BOM — aceitável para detecção e estimativa de pose.")
else:
    print("Avaliação: INSUFICIENTE — revisar captura e padrão de calibração.")


# ============================================================
# ETAPA 4 — Comparação com a matriz intrínseca verdadeira
# ============================================================

print("\n" + "-" * 60)
print("Comparação K estimado vs. K verdadeiro:")
print(f"  fx estimado = {K[0,0]:.2f}  |  fx real = {TRUE_K[0,0]:.2f}")
print(f"  fy estimado = {K[1,1]:.2f}  |  fy real = {TRUE_K[1,1]:.2f}")
print(f"  cx estimado = {K[0,2]:.2f}  |  cx real = {TRUE_K[0,2]:.2f}")
print(f"  cy estimado = {K[1,2]:.2f}  |  cy real = {TRUE_K[1,2]:.2f}")


# ============================================================
# ETAPA 5 — Painel original vs. corrigida (undistort)
# ============================================================

img_original = cv2.imread(str(used[9]))
img_corrigida = cv2.undistort(img_original, K, dist)

painel = np.hstack([img_original, img_corrigida])

cv2.putText(
    painel, "ORIGINAL", (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
)
cv2.putText(
    painel, "CORRIGIDA (undistort)", (img_original.shape[1] + 30, 50),
    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 120, 0), 3
)

separador_x = img_original.shape[1]
cv2.line(
    painel,
    (separador_x, 0),
    (separador_x, painel.shape[0]),
    (255, 255, 255), 2
)

saida = ROOT / "saidas" / "item_a_painel_calibracao.png"
cv2.imwrite(str(saida), painel)
print(f"\nPainel salvo em: {saida}")
