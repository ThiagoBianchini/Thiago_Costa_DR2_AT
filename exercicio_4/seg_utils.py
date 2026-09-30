"""
Módulo utilitário para o Exercício 4 — Segmentação semântica e pipeline integrativo.

Concentra funções de criação de cenas sintéticas externas, segmentação HSV,
paleta de cores, sobreposição de máscaras e medição de tempo.
"""

from pathlib import Path
import cv2
import numpy as np
import time

CLASSES_DIDATICAS = {
    0: "fundo", 1: "ceu", 2: "pista", 3: "calcada",
    4: "vegetacao", 5: "veiculo", 6: "pedestre", 7: "faixa"
}

PALETTE = np.array([
    [0, 0, 0],
    [255, 180, 80],
    [80, 80, 80],
    [160, 80, 160],
    [80, 180, 80],
    [70, 70, 220],
    [220, 60, 60],
    [255, 255, 255]
], dtype=np.uint8)


# ============================================================
# FUNÇÕES DE AMBIENTE
# ============================================================

def ensure(root):
    Path(root, "imagens").mkdir(exist_ok=True, parents=True)
    Path(root, "saidas").mkdir(exist_ok=True, parents=True)


# ============================================================
# CRIAÇÃO DE CENAS EXTERNAS SINTÉTICAS
# ============================================================

def criar_cena_externa(idx=0, w=960, h=540):
    img = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)

    img[:h // 3, :] = (210, 180, 120)
    mask[:h // 3, :] = 1

    img[h // 3:, :] = (90, 90, 90)
    mask[h // 3:, :] = 2

    cv2.rectangle(img, (0, h // 3), (w // 4, h), (170, 110, 90), -1)
    mask[h // 3:, 0:w // 4] = 3

    cv2.rectangle(img, (3 * w // 4, h // 3), (w, h), (170, 110, 90), -1)
    mask[h // 3:, 3 * w // 4:w] = 3

    for x in [120 + idx * 20, 760 - idx * 15, 500 + idx * 5]:
        cv2.circle(img, (x, h // 3 - 15), 55, (60, 160, 60), -1)
        cv2.rectangle(img, (x - 8, h // 3 - 10), (x + 8, h // 3 + 100), (60, 90, 40), -1)
        cv2.circle(mask, (x, h // 3 - 15), 55, 4)
        cv2.rectangle(mask, (x - 8, h // 3 - 10), (x + 8, h // 3 + 100), 4, -1)

    for x, y, ww, hh in [(350 + idx * 12, 360, 130, 60), (560 - idx * 8, 400, 170, 72)]:
        cv2.rectangle(img, (x, y), (x + ww, y + hh), (30, 50, 210), -1)
        cv2.rectangle(mask, (x, y), (x + ww, y + hh), 5, -1)
        cv2.circle(img, (x + 25, y + hh), 20, (20, 20, 20), -1)
        cv2.circle(img, (x + ww - 25, y + hh), 20, (20, 20, 20), -1)

    for x in [240 + idx * 25, 690 - idx * 18]:
        cv2.circle(img, (x, 310), 12, (20, 20, 20), -1)
        cv2.rectangle(img, (x - 8, 322), (x + 8, 370), (30, 70, 220), -1)
        cv2.line(img, (x, 370), (x - 16, 415), (30, 30, 30), 5)
        cv2.line(img, (x, 370), (x + 16, 415), (30, 30, 30), 5)
        cv2.rectangle(mask, (x - 18, 295), (x + 18, 420), 6, -1)

    for y in range(390, 520, 55):
        cv2.rectangle(img, (w // 2 - 25, y), (w // 2 + 25, y + 28), (255, 255, 255), -1)
        cv2.rectangle(mask, (w // 2 - 25, y), (w // 2 + 25, y + 28), 7, -1)

    return img, mask


def salvar_cenas(root, n=5):
    ensure(root)
    paths = []
    for i in range(n):
        img, mask = criar_cena_externa(i)
        p = Path(root, "imagens", f"cena_{i + 1:02d}.png")
        m = Path(root, "imagens", f"cena_{i + 1:02d}_mask.png")
        cv2.imwrite(str(p), img)
        cv2.imwrite(str(m), mask)
        paths.append(p)
    return paths


# ============================================================
# FUNÇÕES DE VISUALIZAÇÃO E CÁLCULO
# ============================================================

def colorir_mascara(mask):
    return PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]


def sobrepor(img, mask, alpha=0.45):
    color = colorir_mascara(mask)
    return cv2.addWeighted(img, 1 - alpha, color, alpha, 0)


def porcentagens(mask, classes=CLASSES_DIDATICAS):
    total = mask.size
    out = {}
    for k, nome in classes.items():
        pct = 100 * float(np.sum(mask == k)) / total
        if pct > 0.01:
            out[nome] = pct
    return out


def hsv_pista(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 40])
    upper = np.array([179, 60, 150])
    return cv2.inRange(hsv, lower, upper)


def painel_lado_a_lado(*imgs):
    return np.hstack(list(imgs))


def cronometro():
    return time.perf_counter()


def ms(t0):
    return (time.perf_counter() - t0) * 1000
