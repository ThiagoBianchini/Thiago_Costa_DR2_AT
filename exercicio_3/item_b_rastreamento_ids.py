"""
Exercício 3 — Item B
Rastreamento com ID persistente, trilhas dos últimos 30 frames,
contagem cumulativa de entradas/saídas e taxa de ID switches.
"""

import time
import cv2
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vision_utils import (
    ensure_dirs,
    draw_scene,
    fake_detector,
    apply_nms,
    draw_detections,
    IoUTracker,
    VIDEO_SIZE,
    FPS_VIDEO
)

ensure_dirs(ROOT)

N_FRAMES = 120


# ============================================================
# ETAPA 1 — Inicialização do tracker e detector
# ============================================================

yolo_disponivel = False
try:
    from ultralytics import YOLO
    model_yolo = YOLO("yolov8n.pt")
    yolo_disponivel = True
    print("YOLOv8n carregado. Usando detector real.")
except ImportError:
    print("Ultralytics não instalada. Usando detector sintético.")

tracker = IoUTracker(iou_thr=0.30, max_missing=8, trail_len=30)


# ============================================================
# ETAPA 2 — Definição da linha virtual de contagem
# ============================================================

LINHA_Y = VIDEO_SIZE[1] // 2
objetos_acima = set()
objetos_abaixo = set()
contagem_entrada = 0
contagem_saida = 0
ids_vistos = set()


# ============================================================
# ETAPA 3 — Processamento frame a frame com rastreamento
# ============================================================

saida_video = ROOT / "saidas" / "item_b_tracking_ids.avi"
writer = cv2.VideoWriter(
    str(saida_video),
    cv2.VideoWriter_fourcc(*"MJPG"),
    FPS_VIDEO,
    VIDEO_SIZE
)

latencias = []
ids_por_frame = []

print("\nProcessando vídeo com rastreamento...")
print("-" * 60)

for i in range(N_FRAMES):
    t0 = time.perf_counter()
    frame = draw_scene(i)

    if yolo_disponivel:
        results = model_yolo(frame, conf=0.25, iou=0.40, verbose=False)[0]
        dets = []
        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            cls_id = int(b.cls[0])
            dets.append({
                "cls": model_yolo.names[cls_id],
                "conf": float(b.conf[0]),
                "bbox": [x1, y1, x2, y2]
            })
    else:
        dets = fake_detector(i)

    dets = apply_nms(dets, 0.25, 0.40)

    tracked = tracker.update(dets)

    # --- Contagem de entradas e saídas pela linha virtual ---
    for d in tracked:
        tid = d["id"]
        x1, y1, x2, y2 = d["bbox"]
        cy = (y1 + y2) // 2

        if tid not in ids_vistos:
            ids_vistos.add(tid)
            contagem_entrada += 1

        if cy < LINHA_Y:
            if tid in objetos_abaixo:
                objetos_abaixo.discard(tid)
                contagem_saida += 1
            objetos_acima.add(tid)
        else:
            if tid in objetos_acima:
                objetos_acima.discard(tid)
            objetos_abaixo.add(tid)

    latencias.append((time.perf_counter() - t0) * 1000)
    ids_por_frame.append(set(d["id"] for d in tracked))

    # --- Desenho do frame anotado ---
    out = draw_detections(frame, tracked, show_id=True, tracks=tracker.trails)

    cv2.line(out, (0, LINHA_Y), (VIDEO_SIZE[0], LINHA_Y), (0, 255, 0), 2)
    cv2.putText(
        out, "LINHA DE CONTAGEM", (10, LINHA_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
    )

    info_text = f"Entradas: {contagem_entrada} | Saidas: {contagem_saida} | IDs ativos: {len(tracked)}"
    cv2.putText(out, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    writer.write(out)

writer.release()


# ============================================================
# ETAPA 4 — Cálculo de ID switches
# ============================================================

id_switches = 0
for j in range(1, len(ids_por_frame)):
    ids_prev = ids_por_frame[j - 1]
    ids_curr = ids_por_frame[j]
    novos = ids_curr - ids_prev
    perdidos = ids_prev - ids_curr
    id_switches += len(novos & set(range(1, tracker.next_id)))

duracao_segundos = N_FRAMES / FPS_VIDEO
duracao_minutos = duracao_segundos / 60
switches_por_minuto = tracker.total_created / duracao_minutos if duracao_minutos > 0 else 0


# ============================================================
# ETAPA 5 — Resultados no terminal
# ============================================================

print("\n" + "=" * 60)
print("RESULTADOS DO RASTREAMENTO")
print("=" * 60)
print(f"Total de frames processados: {N_FRAMES}")
print(f"Duração do vídeo:            {duracao_segundos:.1f} s ({duracao_minutos:.2f} min)")
print(f"Latência média por frame:    {np.mean(latencias):.2f} ms")
print(f"FPS médio estimado:          {1000 / np.mean(latencias):.1f}")
print(f"")
print(f"Total de IDs criados:        {tracker.total_created}")
print(f"Total de IDs removidos:      {tracker.total_removed}")
print(f"Contagem de entradas:        {contagem_entrada}")
print(f"Contagem de saídas:          {contagem_saida}")
print(f"Taxa de ID switches/min:     {switches_por_minuto:.1f}")
print(f"")
print(f"Vídeo salvo em: {saida_video}")


# ============================================================
# ETAPA 6 — Discussão ética sobre contagem por drone
# ============================================================

print("\n" + "=" * 60)
print("DISCUSSÃO ÉTICA — Contagem de pedestres por drone")
print("=" * 60)
print("""
Este sistema de rastreamento com IDs persistentes poderia ser utilizado
para contagem de pedestres em drones de vigilância urbana. No entanto,
há considerações éticas importantes:

1. PRIVACIDADE: A captura de imagens em espaço público sem consentimento
   pode violar direitos de privacidade (LGPD no Brasil, GDPR na Europa).
   Deve-se anonimizar as imagens e não armazenar dados biométricos.

2. VIGILÂNCIA EM MASSA: O uso indiscriminado de drones para monitoramento
   pode configurar vigilância em massa, com impacto desproporcional sobre
   comunidades vulneráveis e minorias.

3. VIÉS ALGORÍTMICO: Detectores treinados com datasets desbalanceados
   podem ter taxa de erro desigual entre grupos demográficos, gerando
   subcontagem ou falsos positivos discriminatórios.

4. TRANSPARÊNCIA: A população monitorada deve ser informada sobre a
   presença do sistema, sua finalidade e os dados coletados.

5. PROPORCIONALIDADE: O uso deve ser proporcional à necessidade, com
   justificativa clara (gestão de tráfego, emergências, planejamento
   urbano) e não para fins de controle social.

Recomendação: limitar o sistema à contagem agregada sem identificação
individual, com dados processados em tempo real e descartados após
a geração de estatísticas anônimas.
""")
