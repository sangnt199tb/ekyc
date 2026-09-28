"""
Engine OCR kết hợp:
- RapidOCR (model PP-OCR của PaddleOCR chạy bằng ONNX Runtime): phát hiện vị trí các dòng chữ
  và đọc chính xác chữ số / tiếng Anh / dải MRZ.
- VietOCR: đọc lại từng dòng để ra tiếng Việt có dấu.
"""
import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# Đường dẫn file weights VietOCR (Docker tải sẵn vào đây). Nếu không có sẽ tự tải về thư mục temp.
VIETOCR_WEIGHTS = os.getenv("VIETOCR_WEIGHTS", "models/vgg_transformer.pth")

MRZ_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")

# Ảnh điện thoại thường rất lớn, thu nhỏ trước khi OCR cho nhanh (chữ trên thẻ vẫn đủ rõ)
MAX_SIDE = 1600


def _resize(img: np.ndarray, max_side: int) -> np.ndarray:
    scale = max_side / max(img.shape[:2])
    if scale >= 1:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


@dataclass
class TextLine:
    text: str       # Kết quả VietOCR (tiếng Việt có dấu)
    raw: str        # Kết quả RapidOCR (tốt cho số, tiếng Anh, MRZ)
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


def looks_like_mrz(raw: str) -> bool:
    s = raw.replace(" ", "").upper()
    return len(s) >= 20 and ("<" in s or s.startswith("IDVNM")) and sum(c in MRZ_CHARS for c in s) / len(s) > 0.9


class OcrEngine:
    _instance = None
    _init_lock = threading.Lock()

    def __init__(self):
        from rapidocr import RapidOCR
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        self.detector = RapidOCR(params={"Global.log_level": "error"})

        cfg = Cfg.load_config_from_name("vgg_transformer")
        if os.path.exists(VIETOCR_WEIGHTS):
            cfg["weights"] = VIETOCR_WEIGHTS
        cfg["device"] = "cpu"
        cfg["cnn"]["pretrained"] = False
        cfg["predictor"]["beamsearch"] = False
        self.recognizer = Predictor(cfg)

        # Model không thread-safe, mỗi lần chỉ xử lý 1 ảnh
        self.lock = threading.Lock()

    @classmethod
    def get(cls) -> "OcrEngine":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = OcrEngine()
        return cls._instance

    def is_vertical(self, img: np.ndarray) -> bool:
        """Ảnh bị xoay 90 độ: phần lớn các dòng chữ phát hiện được cao hơn rộng."""
        small = _resize(img, 960)
        with self.lock:
            # RapidOCR ghi nhớ tham số giữa các lần gọi nên luôn truyền đủ use_det/use_cls/use_rec
            result = self.detector(small, use_det=True, use_cls=False, use_rec=False)
        if result.boxes is None or len(result.boxes) == 0:
            return False
        boxes = np.array(result.boxes)
        heights = boxes[:, :, 1].max(axis=1) - boxes[:, :, 1].min(axis=1)
        widths = boxes[:, :, 0].max(axis=1) - boxes[:, :, 0].min(axis=1)
        return (heights > widths).sum() > len(boxes) / 2

    def read_lines(self, img: np.ndarray) -> list[TextLine]:
        from rapidocr.utils.process_img import get_rotate_crop_image

        img = _resize(img, MAX_SIDE)
        with self.lock:
            # Tắt cls để ảnh lộn ngược cho ra chữ vô nghĩa -> reader nhận biết và lật lại ảnh
            result = self.detector(img, use_det=True, use_cls=False, use_rec=True)
            if result.boxes is None:
                return []

            lines, crops = [], []
            for box, raw in zip(result.boxes, result.txts):
                pts = np.array(box, dtype=np.float32)
                lines.append(TextLine("", raw, pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
                crop = get_rotate_crop_image(img, pts)
                crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

            # MRZ chỉ gồm chữ in hoa/số/'<' -> RapidOCR đọc tốt hơn, không cần VietOCR
            idx = [i for i, ln in enumerate(lines) if not looks_like_mrz(ln.raw)]
            if idx:
                texts = self.recognizer.predict_batch([crops[i] for i in idx])
                for i, t in zip(idx, texts):
                    lines[i].text = t
            for ln in lines:
                if not ln.text:
                    ln.text = ln.raw

        lines.sort(key=lambda ln: (ln.y0, ln.x0))
        return lines
