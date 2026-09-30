"""
Módulo utilitário para o Exercício 2 — OpenCV DNN e pipeline integrado.

Concentra funções de criação de dados sintéticos, inferência DNN,
segmentação HSV, features ORB e detecção clássica HOG/Haar.
"""

from pathlib import Path
import csv
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_CLASS = ROOT / "data" / "classificacao"
DATA_PIPE = ROOT / "data" / "pipeline"
MODELS = ROOT / "modelos"
OUT = ROOT / "saidas"

IMAGE_SIZE = (224, 224)

DEFAULT_K = np.array(
    [[850.0, 0.0, 320.0],
     [0.0, 850.0, 240.0],
     [0.0, 0.0, 1.0]],
    dtype=np.float32
)

DEFAULT_DIST = np.array(
    [[-0.20, 0.08, 0.0, 0.0, 0.0]],
    dtype=np.float32
)


# ============================================================
# CRIAÇÃO DE DIRETÓRIOS E DADOS SINTÉTICOS
# ============================================================

def ensure_dirs():
    for p in [DATA_CLASS, DATA_PIPE, MODELS, OUT]:
        p.mkdir(parents=True, exist_ok=True)


def create_synthetic_classification_images(n=10):
    ensure_dirs()
    names = [
        "carro", "caneca", "bola", "livro", "garrafa",
        "teclado", "robo", "caixa", "ferramenta", "capacete"
    ]
    colors = [
        (230, 230, 230), (245, 245, 255), (255, 245, 240),
        (240, 255, 245), (245, 240, 255), (255, 255, 235),
        (235, 255, 255), (245, 245, 245), (255, 240, 240),
        (240, 240, 255)
    ]
    paths = []
    for i, name in enumerate(names[:n]):
        img = np.full((360, 480, 3), colors[i], np.uint8)
        cv2.rectangle(img, (80, 80), (400, 280), (40 + i * 12, 80 + i * 8, 160 + i * 5), 3)
        cv2.circle(img, (240, 180), 55 + i * 2, (80 + i * 10, 50 + i * 8, 200 - i * 5), -1)
        cv2.putText(img, name, (95, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
        path = DATA_CLASS / f"{i + 1:02d}_{name}.jpg"
        cv2.imwrite(str(path), img)
        paths.append(path)

    csv_path = ROOT / "data" / "labels_top1.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "label_esperado"])
        for p in paths:
            w.writerow([p.name, ""])
    return paths


def list_classification_images():
    ensure_dirs()
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    paths = []
    for e in exts:
        paths.extend(DATA_CLASS.glob(e))
    paths = sorted(paths)
    if not paths:
        paths = create_synthetic_classification_images(10)
    return paths


def create_pipeline_frame():
    ensure_dirs()
    img = np.full((480, 640, 3), (235, 235, 235), np.uint8)
    cv2.rectangle(img, (35, 35), (605, 445), (210, 210, 210), 2)
    cv2.rectangle(img, (120, 120), (380, 340), (0, 0, 255), -1)
    cv2.rectangle(img, (150, 150), (350, 310), (255, 255, 255), 3)
    for x in range(160, 340, 35):
        cv2.circle(img, (x, 190), 8, (0, 0, 0), -1)
        cv2.line(img, (x, 230), (x + 20, 260), (0, 0, 0), 2)
    cv2.putText(img, "ROI HSV + ORB + DNN", (95, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    cv2.imwrite(str(DATA_PIPE / "frame_sintetico.jpg"), img)
    return DATA_PIPE / "frame_sintetico.jpg"


def read_pipeline_frame():
    real = DATA_PIPE / "frame_real.jpg"
    if real.exists():
        return cv2.imread(str(real)), real
    p = create_pipeline_frame()
    return cv2.imread(str(p)), p


# ============================================================
# FUNÇÕES DE INFERÊNCIA DNN
# ============================================================

def squeezenet_blob(img):
    return cv2.dnn.blobFromImage(
        img,
        scalefactor=1.0,
        size=(227, 227),
        mean=(104.0, 117.0, 123.0),
        swapRB=False,
        crop=False
    )


def mobilenet_blob_bgr(img):
    return cv2.dnn.blobFromImage(
        img,
        scalefactor=1.0 / 127.5,
        size=IMAGE_SIZE,
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False
    )


def softmax(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def load_labels():
    label_file = MODELS / "imagenet_labels.txt"
    if label_file.exists():
        labels = [
            line.strip()
            for line in label_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(labels) >= 1000:
            return labels[:1000]
    return [f"classe_{i:03d}" for i in range(1000)]


def get_opencv_net():
    sq_model = MODELS / "squeezenet1.1.onnx"
    if sq_model.exists():
        net = cv2.dnn.readNetFromONNX(str(sq_model))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        return net
    tflite_model = MODELS / "mobilenetv2_imagenet.tflite"
    if tflite_model.exists():
        net = cv2.dnn.readNetFromTFLite(str(tflite_model))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        return net
    raise FileNotFoundError(
        "Nenhum modelo encontrado em modelos/. Baixe squeezenet1.1.onnx ou converta mobilenetv2."
    )


def predict_opencv(img, net=None):
    if net is None:
        net = get_opencv_net()
    sq_model = MODELS / "squeezenet1.1.onnx"
    if sq_model.exists():
        blob = squeezenet_blob(img)
    else:
        blob = mobilenet_blob_bgr(img)
    net.setInput(blob)
    out = net.forward()
    prob = softmax(out)
    return prob



def topk(prob, k=3):
    idx = np.argsort(prob)[::-1][:k]
    return [(int(i), float(prob[i])) for i in idx]


def draw_top3(img, top3, labels=None, x=20, y=35):
    if labels is None:
        labels = load_labels()
    out = img.copy()
    cv2.rectangle(out, (10, 10), (470, 120), (255, 255, 255), -1)
    cv2.rectangle(out, (10, 10), (470, 120), (0, 0, 0), 2)
    cv2.putText(out, "Top-3 DNN", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    for j, (idx, conf) in enumerate(top3, start=1):
        label = labels[idx] if idx < len(labels) else f"classe_{idx}"
        txt = f"{j}) {label}: {conf * 100:.1f}%"
        cv2.putText(out, txt, (x, y + 28 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 180), 2)
    return out


def time_ms(func, loops=10, warmup=3):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(loops):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


# ============================================================
# FUNÇÕES DO PIPELINE INTEGRADO
# ============================================================

def segment_red_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 80, 70])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 70])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.medianBlur(mask, 5)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:
        return mask, None
    x, y, w, h = cv2.boundingRect(c)
    return mask, (x, y, w, h)


def orb_features(img, max_features=250):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    orb = cv2.ORB_create(nfeatures=max_features)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des


def hog_or_haar_detector(frame):
    boxes = []
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(
        frame, winStride=(8, 8), padding=(8, 8), scale=1.05
    )
    for (x, y, w, h) in rects:
        boxes.append(("HOG", x, y, w, h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face = cv2.CascadeClassifier(cascade_path)
    faces = face.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        boxes.append(("Haar", x, y, w, h))
    return boxes
