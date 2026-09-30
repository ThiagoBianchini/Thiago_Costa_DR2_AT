"""
Módulo utilitário para o Exercício 3 — Detecção e rastreamento.

Concentra funções de vídeo sintético, detecção simulada, NMS,
IoU, rastreamento com ID persistente e visualização de trilhas.
"""

from pathlib import Path
from collections import deque
import time
import csv
import cv2
import numpy as np

CLASSES = ["person", "car", "bicycle", "bus"]

COLORS = {
    "person":  (255, 120, 0),
    "car":     (0, 180, 255),
    "bicycle": (255, 0, 180),
    "bus":     (0, 80, 255),
}

VIDEO_SIZE = (960, 540)
FPS_VIDEO = 20


# ============================================================
# FUNÇÕES DE AMBIENTE
# ============================================================

def ensure_dirs(root):
    root = Path(root)
    for name in ["dados", "saidas", "modelos", "relatorios"]:
        (root / name).mkdir(exist_ok=True, parents=True)


# ============================================================
# GERAÇÃO DE CENA E VÍDEO SINTÉTICOS
# ============================================================

def synthetic_objects(frame_idx):
    objs = []

    x = 40 + frame_idx * 5
    y = 280 + int(15 * np.sin(frame_idx / 8))
    objs.append({"cls": "person", "conf": 0.82, "bbox": [x, y, x + 48, y + 110]})

    x2 = 850 - frame_idx * 4
    y2 = 345
    objs.append({"cls": "car", "conf": 0.88, "bbox": [x2, y2, x2 + 120, y2 + 60]})

    x3 = 160 + frame_idx * 3
    y3 = 95 + frame_idx * 2
    objs.append({"cls": "bicycle", "conf": 0.74, "bbox": [x3, y3, x3 + 85, y3 + 70]})

    if frame_idx > 35:
        x4 = 720 - (frame_idx - 35) * 2
        y4 = 165
        objs.append({"cls": "bus", "conf": 0.79, "bbox": [x4, y4, x4 + 150, y4 + 85]})

    valid = []
    for o in objs:
        x1, y1, x2, y2 = o["bbox"]
        if x2 > 0 and y2 > 0 and x1 < VIDEO_SIZE[0] and y1 < VIDEO_SIZE[1]:
            o["bbox"] = [
                max(0, x1), max(0, y1),
                min(VIDEO_SIZE[0] - 1, x2), min(VIDEO_SIZE[1] - 1, y2)
            ]
            valid.append(o)
    return valid


def draw_scene(frame_idx):
    w, h = VIDEO_SIZE
    frame = np.full((h, w, 3), (35, 40, 45), dtype=np.uint8)

    cv2.rectangle(frame, (0, 0), (w, 150), (80, 100, 120), -1)
    cv2.rectangle(frame, (0, 150), (w, h), (55, 55, 55), -1)
    cv2.rectangle(frame, (0, 150), (w, 205), (90, 90, 85), -1)
    cv2.rectangle(frame, (0, 440), (w, h), (90, 90, 85), -1)

    for x in range(-100, w + 100, 160):
        cv2.line(
            frame,
            (x + frame_idx * 3 % 160, 300),
            (x + 70 + frame_idx * 3 % 160, 300),
            (220, 220, 220), 4
        )
    cv2.line(frame, (0, 270), (w, 270), (180, 180, 180), 1)
    cv2.line(frame, (0, 365), (w, 365), (180, 180, 180), 1)

    for i, x in enumerate(range(20, w, 115)):
        cv2.rectangle(frame, (x, 30), (x + 65, 150), (70 + i * 6 % 60, 75, 82), -1)
        for wx in range(x + 10, x + 60, 20):
            for wy in range(45, 130, 25):
                cv2.rectangle(frame, (wx, wy), (wx + 8, wy + 10), (160, 180, 190), -1)

    for obj in synthetic_objects(frame_idx):
        x1, y1, x2, y2 = map(int, obj["bbox"])
        color = COLORS[obj["cls"]]
        if obj["cls"] == "person":
            cv2.circle(frame, ((x1 + x2) // 2, y1 + 18), 16, color, -1)
            cv2.rectangle(frame, (x1 + 12, y1 + 35), (x2 - 12, y2 - 25), color, -1)
            cv2.line(frame, (x1 + 18, y2 - 25), (x1 + 10, y2), color, 6)
            cv2.line(frame, (x2 - 18, y2 - 25), (x2 - 10, y2), color, 6)
        elif obj["cls"] == "car":
            cv2.rectangle(frame, (x1, y1 + 18), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1 + 25, y1), (x2 - 25, y1 + 26), color, -1)
            cv2.circle(frame, (x1 + 25, y2), 12, (10, 10, 10), -1)
            cv2.circle(frame, (x2 - 25, y2), 12, (10, 10, 10), -1)
        elif obj["cls"] == "bicycle":
            cv2.circle(frame, (x1 + 18, y2 - 15), 17, color, 3)
            cv2.circle(frame, (x2 - 18, y2 - 15), 17, color, 3)
            cv2.line(frame, (x1 + 18, y2 - 15), (x1 + 45, y1 + 20), color, 4)
            cv2.line(frame, (x1 + 45, y1 + 20), (x2 - 18, y2 - 15), color, 4)
            cv2.line(frame, (x1 + 45, y1 + 20), (x2 - 35, y1 + 15), color, 4)
        elif obj["cls"] == "bus":
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            for wx in range(x1 + 15, x2 - 20, 35):
                cv2.rectangle(frame, (wx, y1 + 15), (wx + 22, y1 + 35), (180, 220, 230), -1)
            cv2.circle(frame, (x1 + 30, y2), 13, (10, 10, 10), -1)
            cv2.circle(frame, (x2 - 30, y2), 13, (10, 10, 10), -1)
    return frame


def create_synthetic_video(path, n_frames=120):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, FPS_VIDEO, VIDEO_SIZE)
    for i in range(n_frames):
        writer.write(draw_scene(i))
    writer.release()
    return path


# ============================================================
# DETECÇÃO SIMULADA E NMS
# ============================================================

def fake_detector(frame_idx, duplicate=True, jitter=3):
    rng = np.random.default_rng(seed=1000 + frame_idx)
    dets = []
    for o in synthetic_objects(frame_idx):
        x1, y1, x2, y2 = o["bbox"]
        noise = rng.integers(-jitter, jitter + 1, size=4)
        b = [x1 + int(noise[0]), y1 + int(noise[1]), x2 + int(noise[2]), y2 + int(noise[3])]
        dets.append({"cls": o["cls"], "conf": float(o["conf"]), "bbox": b})
        if duplicate:
            b2 = [b[0] + 5, b[1] + 3, b[2] + 5, b[3] + 3]
            dets.append({"cls": o["cls"], "conf": float(o["conf"] - 0.10), "bbox": b2})
    return dets


def bbox_to_xywh(box):
    x1, y1, x2, y2 = box
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


def apply_nms(detections, score_thr=0.25, nms_thr=0.40):
    boxes = [bbox_to_xywh(d["bbox"]) for d in detections]
    scores = [float(d["conf"]) for d in detections]
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_thr, nms_thr)
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).flatten().tolist()
    return [detections[i] for i in idxs]


# ============================================================
# VISUALIZAÇÃO
# ============================================================

def draw_detections(frame, detections, show_id=False, tracks=None):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cls = d.get("cls", "obj")
        conf = d.get("conf", 0.0)
        color = COLORS.get(cls, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f}"
        if show_id and "id" in d:
            label = f"ID {d['id']} | " + label
        cv2.putText(out, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if tracks:
        for tid, pts in tracks.items():
            pts = list(pts)
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 255), 2)
    return out


# ============================================================
# IoU E RASTREAMENTO
# ============================================================

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union == 0 else inter / union


class IoUTracker:
    def __init__(self, iou_thr=0.30, max_missing=8, trail_len=30):
        self.iou_thr = iou_thr
        self.max_missing = max_missing
        self.next_id = 1
        self.tracks = {}
        self.trails = {}
        self.id_switches = 0
        self.total_created = 0
        self.total_removed = 0
        self.trail_len = trail_len

    def update(self, detections):
        assigned_tracks = set()
        assigned_dets = set()
        results = []

        track_ids = list(self.tracks.keys())
        pairs = []
        for tid in track_ids:
            for di, det in enumerate(detections):
                pairs.append((iou(self.tracks[tid]["bbox"], det["bbox"]), tid, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for score, tid, di in pairs:
            if score < self.iou_thr:
                continue
            if tid in assigned_tracks or di in assigned_dets:
                continue
            self.tracks[tid].update({
                "bbox": detections[di]["bbox"],
                "cls": detections[di]["cls"],
                "conf": detections[di]["conf"],
                "missing": 0
            })
            assigned_tracks.add(tid)
            assigned_dets.add(di)

        for di, det in enumerate(detections):
            if di not in assigned_dets:
                tid = self.next_id
                self.next_id += 1
                self.total_created += 1
                self.tracks[tid] = {
                    "bbox": det["bbox"],
                    "cls": det["cls"],
                    "conf": det["conf"],
                    "missing": 0
                }
                self.trails[tid] = deque(maxlen=self.trail_len)
                assigned_tracks.add(tid)

        to_delete = []
        for tid in list(self.tracks.keys()):
            if tid not in assigned_tracks:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    to_delete.append(tid)
            else:
                x1, y1, x2, y2 = self.tracks[tid]["bbox"]
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                self.trails.setdefault(tid, deque(maxlen=self.trail_len)).append(center)

        for tid in to_delete:
            self.tracks.pop(tid, None)
            self.total_removed += 1

        for tid, tr in self.tracks.items():
            d = {"id": tid, "bbox": tr["bbox"], "cls": tr["cls"], "conf": tr["conf"]}
            results.append(d)
        return results


# ============================================================
# UTILITÁRIOS
# ============================================================

def write_csv(path, rows, headers):
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def ms(self):
        return (time.perf_counter() - self.t0) * 1000
