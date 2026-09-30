"""
Exercício 3 — Item A
Detecção de objetos em tempo real com YOLOv8n e SSD MobileNetV2:
processamento de vídeo sintético com bounding boxes, NMS,
medição de FPS/latência e tabela comparativa.
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
    VIDEO_SIZE,
    FPS_VIDEO,
    create_synthetic_video
)

ensure_dirs(ROOT)

N_FRAMES = 120


# ============================================================
# ETAPA 1 — Geração do vídeo sintético de teste
# ============================================================

video_path = ROOT / "dados" / "video_sintetico.avi"
create_synthetic_video(video_path, N_FRAMES)
print(f"Vídeo sintético criado: {video_path}")


# ============================================================
# ETAPA 2 — Detecção com YOLOv8n (Ultralytics)
# ============================================================

yolo_disponivel = True
try:
    from ultralytics import YOLO
    model_yolo = YOLO("yolov8n.pt")
    print("\nYOLOv8n carregado com sucesso via Ultralytics.")
except ImportError:
    yolo_disponivel = False
    print("\nUltralytics não instalada. Usando detector sintético para YOLO.")

saida_yolo = ROOT / "saidas" / "item_a_yolo_deteccao.avi"
writer_yolo = cv2.VideoWriter(
    str(saida_yolo),
    cv2.VideoWriter_fourcc(*"MJPG"),
    FPS_VIDEO,
    VIDEO_SIZE
)

latencias_yolo = []

print("\nProcessando com YOLO...")
for i in range(N_FRAMES):
    frame = draw_scene(i)
    t0 = time.perf_counter()

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

    latencias_yolo.append((time.perf_counter() - t0) * 1000)
    out = draw_detections(frame, dets)
    writer_yolo.write(out)

writer_yolo.release()

fps_yolo = 1000 / np.mean(latencias_yolo)
lat_yolo = np.mean(latencias_yolo)
print(f"YOLO — Latência média: {lat_yolo:.2f} ms | FPS: {fps_yolo:.1f}")


# ============================================================
# ETAPA 3 — Detecção com SSD MobileNetV2 (OpenCV DNN)
# ============================================================

ssd_disponivel = False
ssd_config = ROOT / "modelos" / "ssd_mobilenet_v2_coco.pbtxt"
ssd_model = ROOT / "modelos" / "frozen_inference_graph.pb"

if ssd_config.exists() and ssd_model.exists():
    ssd_net = cv2.dnn.readNetFromTensorflow(str(ssd_model), str(ssd_config))
    ssd_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    ssd_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    ssd_disponivel = True
    print("\nSSD MobileNetV2 carregado via OpenCV DNN.")
else:
    print("\nModelo SSD não encontrado. Usando detector sintético para SSD.")

saida_ssd = ROOT / "saidas" / "item_a_ssd_deteccao.avi"
writer_ssd = cv2.VideoWriter(
    str(saida_ssd),
    cv2.VideoWriter_fourcc(*"MJPG"),
    FPS_VIDEO,
    VIDEO_SIZE
)

latencias_ssd = []
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
    5: "airplane", 6: "bus", 7: "train", 8: "truck"
}

print("Processando com SSD...")
for i in range(N_FRAMES):
    frame = draw_scene(i)
    t0 = time.perf_counter()

    if ssd_disponivel:
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), True)
        ssd_net.setInput(blob)
        output = ssd_net.forward()
        dets = []
        for det in output[0, 0]:
            conf = float(det[2])
            if conf < 0.25:
                continue
            cls_id = int(det[1])
            cls_name = COCO_LABELS.get(cls_id, f"classe_{cls_id}")
            h_frame, w_frame = frame.shape[:2]
            x1 = int(det[3] * w_frame)
            y1 = int(det[4] * h_frame)
            x2 = int(det[5] * w_frame)
            y2 = int(det[6] * h_frame)
            dets.append({"cls": cls_name, "conf": conf, "bbox": [x1, y1, x2, y2]})
    else:
        dets = fake_detector(i, duplicate=False)

    latencias_ssd.append((time.perf_counter() - t0) * 1000)
    out = draw_detections(frame, dets)
    writer_ssd.write(out)

writer_ssd.release()

fps_ssd = 1000 / np.mean(latencias_ssd)
lat_ssd = np.mean(latencias_ssd)
print(f"SSD  — Latência média: {lat_ssd:.2f} ms | FPS: {fps_ssd:.1f}")


# ============================================================
# ETAPA 4 — Tabela comparativa
# ============================================================

yolo_params = 3_200_000
yolo_size = 6.3
ssd_params = 4_300_000
ssd_size = 18.0

if yolo_disponivel:
    yolo_model_path = Path("yolov8n.pt")
    if yolo_model_path.exists():
        yolo_size = yolo_model_path.stat().st_size / (1024 * 1024)

print("\n" + "=" * 70)
print("TABELA COMPARATIVA — YOLO vs. SSD")
print("=" * 70)
print(f"{'Métrica':<25} {'YOLOv8n':>15} {'SSD MobileNetV2':>18}")
print("-" * 70)
print(f"{'FPS médio':<25} {fps_yolo:>15.1f} {fps_ssd:>18.1f}")
print(f"{'Latência média (ms)':<25} {lat_yolo:>15.2f} {lat_ssd:>18.2f}")
print(f"{'Parâmetros (estimado)':<25} {f'~{yolo_params/1e6:.1f}M':>15} {f'~{ssd_params/1e6:.1f}M':>18}")
print(f"{'Tamanho em disco (MB)':<25} {yolo_size:>15.1f} {ssd_size:>18.1f}")
print("-" * 70)


# ============================================================
# ETAPA 5 — Conclusão para robótica embarcada
# ============================================================

print("\nCONCLUSÃO PARA ROBÓTICA EMBARCADA:")
print("-" * 70)

if fps_yolo > fps_ssd:
    melhor_fps = "YOLOv8n"
else:
    melhor_fps = "SSD MobileNetV2"

if yolo_size < ssd_size:
    menor_modelo = "YOLOv8n"
else:
    menor_modelo = "SSD MobileNetV2"

print(f"Modelo com maior FPS: {melhor_fps}")
print(f"Modelo com menor tamanho em disco: {menor_modelo}")
print()
print("Para robótica embarcada com restrição de 5W (ex: Jetson Nano, RPi):")
print("- YOLOv8n é preferível por seu menor tamanho em disco (~6 MB) e")
print("  arquitetura otimizada para inferência eficiente.")
print("- SSD MobileNetV2 pode ser vantajoso em pipelines já baseados em")
print("  TensorFlow Lite com aceleração por NPU/TPU.")
print("- A decisão final depende do hardware alvo, precisão mínima exigida")
print("  e latência máxima tolerável pela aplicação.")
