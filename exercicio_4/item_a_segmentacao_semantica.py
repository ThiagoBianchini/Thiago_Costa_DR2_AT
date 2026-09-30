"""
Exercício 4 — Item A
Segmentação semântica com FCN-ResNet50, comparação com segmentação HSV,
sobreposição de máscara semitransparente e cálculo de área por classe.
"""

from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from seg_utils import (
    salvar_cenas,
    colorir_mascara,
    sobrepor,
    porcentagens,
    hsv_pista,
    painel_lado_a_lado,
    CLASSES_DIDATICAS,
    ensure
)

ensure(ROOT)


# ============================================================
# ETAPA 1 — Geração de 5 cenas externas sintéticas
# ============================================================

paths = salvar_cenas(ROOT, 5)
print(f"Cenas sintéticas geradas: {len(paths)}")


# ============================================================
# ETAPA 2 — Carregamento do modelo FCN-ResNet50
# ============================================================

import torch
from PIL import Image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights).eval()
preprocess = weights.transforms()

PASCAL_VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]

rng = np.random.default_rng(42)
palette_fcn = rng.integers(0, 255, (256, 3), dtype=np.uint8)
palette_fcn[0] = [0, 0, 0]

print("FCN-ResNet50 carregado com sucesso.\n")


# ============================================================
# ETAPA 3 — Processamento das 5 imagens com segmentação semântica
# ============================================================

for p in paths:
    print(f"Processando: {p.name}")
    t0 = time.perf_counter()

    img_bgr = cv2.imread(str(p))
    pil_img = Image.open(p).convert("RGB")
    x = preprocess(pil_img).unsqueeze(0)

    with torch.no_grad():
        output = model(x)["out"][0]
        mask_fcn = output.argmax(0).byte().cpu().numpy()

    tempo_fcn = (time.perf_counter() - t0) * 1000

    # --- Mapa de segmentação colorido ---
    color_map = cv2.resize(palette_fcn[mask_fcn], (img_bgr.shape[1], img_bgr.shape[0]))

    # --- Sobreposição semitransparente ---
    overlay = cv2.addWeighted(img_bgr, 0.60, color_map, 0.40, 0)
    saida_overlay = ROOT / "saidas" / f"item_a_overlay_{p.stem}.png"
    cv2.imwrite(str(saida_overlay), overlay)

    # --- Porcentagem de área por classe ---
    mask_resized = cv2.resize(mask_fcn, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    total_pixels = mask_resized.size
    classes_encontradas = np.unique(mask_resized)

    print(f"  Tempo de inferência: {tempo_fcn:.1f} ms")
    print(f"  Classes detectadas:")
    for cls_id in classes_encontradas:
        cls_name = PASCAL_VOC_CLASSES[cls_id] if cls_id < len(PASCAL_VOC_CLASSES) else f"classe_{cls_id}"
        pct = 100 * np.sum(mask_resized == cls_id) / total_pixels
        if pct > 0.1:
            print(f"    {cls_name:>15s}: {pct:6.2f}%")
    print()


# ============================================================
# ETAPA 4 — Comparação visual: segmentação semântica vs. HSV
# ============================================================

print("Gerando painéis comparativos semântica vs. HSV...\n")

for p in paths:
    img_bgr = cv2.imread(str(p))

    # --- Segmentação HSV ---
    mask_hsv = hsv_pista(img_bgr)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)

    hsv_color = cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR)
    overlay_hsv = cv2.addWeighted(img_bgr, 0.70, hsv_color, 0.30, 0)

    # --- Segmentação semântica (recarregar overlay já salvo) ---
    overlay_sem = cv2.imread(str(ROOT / "saidas" / f"item_a_overlay_{p.stem}.png"))

    # --- Painel lado a lado ---
    cv2.putText(img_bgr, "ORIGINAL", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(overlay_hsv, "HSV", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(overlay_sem, "FCN-ResNet50", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    painel = painel_lado_a_lado(img_bgr, overlay_hsv, overlay_sem)
    saida_painel = ROOT / "saidas" / f"item_a_painel_{p.stem}.png"
    cv2.imwrite(str(saida_painel), painel)
    print(f"  Painel salvo: {saida_painel.name}")


# ============================================================
# ETAPA 5 — Discussão: vantagens e limitações de cada abordagem
# ============================================================

print("\n" + "=" * 70)
print("DISCUSSÃO — Segmentação Semântica vs. HSV para Veículos Autônomos")
print("=" * 70)
print("""
SEGMENTAÇÃO POR COR (HSV):
  Vantagens:
  - Extremamente rápida (< 5 ms por frame)
  - Não requer GPU nem modelo pré-treinado
  - Baixo consumo de memória e energia
  - Implementação simples e determinística

  Limitações:
  - Depende fortemente das condições de iluminação
  - Não distingue categorias semânticas (pessoa vs. veículo)
  - Requer ajuste manual dos thresholds para cada cena
  - Falha com sombras, reflexos e variações de cor

SEGMENTAÇÃO SEMÂNTICA (FCN-ResNet50):
  Vantagens:
  - Classifica cada pixel por categoria semântica
  - Robusta a variações de iluminação e textura
  - Generaliza para cenas diversas sem ajuste manual
  - Diferencia pedestres, veículos, pista e calçada

  Limitações:
  - Requer GPU para inferência em tempo real
  - Consumo elevado de memória e energia
  - Latência maior (50-200 ms por frame em CPU)
  - Dependência de datasets de treinamento representativos

CONCLUSÃO:
  Para um veículo autônomo urbano, a segmentação semântica é essencial
  para compreensão completa da cena, mas a segmentação HSV pode ser
  usada como pré-filtro rápido em hardware com restrição energética.
  A combinação das duas técnicas permite um pipeline híbrido eficiente.
""")
