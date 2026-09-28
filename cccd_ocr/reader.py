"""
Điều phối OCR 2 mặt CCCD và gộp kết quả từ 3 nguồn theo độ tin cậy:
    QR code (chính xác tuyệt đối) > OCR chữ > MRZ (không dấu, dùng để đối chiếu / bổ sung)
"""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from .engine import OcrEngine, TextLine, looks_like_mrz
from .fields import count_labels, detect_card_type, extract_fields, strip_accents
from .mrz import parse_mrz
from .qr import decode_qr, parse_qr

FIELDS = [
    "id_number", "old_id_number", "full_name", "date_of_birth", "sex", "nationality",
    "place_of_origin", "place_of_residence", "place_of_birth",
    "date_of_issue", "date_of_expiry", "personal_identification", "issuing_authority",
]

ISSUING_AUTHORITY = {
    "CAN_CUOC_2024": "Bộ Công an",
    "CCCD_CHIP_2021": "Cục Cảnh sát quản lý hành chính về trật tự xã hội",
}

# Ưu tiên nguồn cho từng trường
PRIORITY = {
    "id_number": ("qr", "ocr", "mrz"),
    "full_name": ("qr", "ocr"),
    "date_of_birth": ("qr", "ocr", "mrz"),
    "sex": ("qr", "ocr", "mrz"),
    "place_of_residence": ("qr", "ocr"),
    "date_of_issue": ("qr", "ocr"),
    "date_of_expiry": ("ocr", "mrz"),
}


def load_image(content: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(content))
        img = ImageOps.exif_transpose(img).convert("RGB")  # Ảnh chụp điện thoại hay bị xoay theo EXIF
    except Exception as e:
        raise ValueError(f"File không phải ảnh hợp lệ: {e}")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _orientation_score(lines: list[TextLine]) -> int:
    return count_labels(lines) + sum(looks_like_mrz(ln.raw) for ln in lines)


def _read_side(engine: OcrEngine, img: np.ndarray) -> tuple[np.ndarray, list[TextLine]]:
    """OCR 1 mặt thẻ, tự xoay ảnh nếu chụp dọc hoặc lộn ngược."""
    if engine.is_vertical(img):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    lines = engine.read_lines(img)
    score = _orientation_score(lines)
    if score >= 2:
        return img, lines

    # Không thấy nhãn nào -> có thể ảnh bị lộn ngược (hoặc xoay 90 độ sai chiều ở bước trên)
    flipped = cv2.rotate(img, cv2.ROTATE_180)
    flipped_lines = engine.read_lines(flipped)
    if _orientation_score(flipped_lines) > score:
        return flipped, flipped_lines
    return img, lines


def _same(a: str, b: str) -> bool:
    return " ".join(strip_accents(a).upper().split()) == " ".join(strip_accents(b).upper().split())


def _check_id_structure(data: dict, warnings: list[str]) -> None:
    """
    Số định danh 12 số: 3 số mã tỉnh + 1 số mã thế kỷ & giới tính + 2 số cuối năm sinh + 6 số ngẫu nhiên.
    Mã thế kỷ/giới tính: 0/1 = Nam/Nữ thế kỷ 20, 2/3 = Nam/Nữ thế kỷ 21...
    """
    id_number, dob, sex = data.get("id_number"), data.get("date_of_birth"), data.get("sex")
    if not id_number or not dob or not sex:
        return
    code = int(id_number[3])
    year = int(dob[-4:])
    expected_century = 1900 + (code // 2) * 100
    expected_sex = "Nam" if code % 2 == 0 else "Nữ"
    if not (expected_century <= year < expected_century + 100) or id_number[4:6] != dob[-2:]:
        warnings.append("Số định danh không khớp với năm sinh")
    if expected_sex != sex:
        warnings.append("Số định danh không khớp với giới tính")


def read_cccd(front_content: bytes, back_content: bytes, debug: bool = False) -> dict:
    engine = OcrEngine.get()
    front_img, front_lines = _read_side(engine, load_image(front_content))
    back_img, back_lines = _read_side(engine, load_image(back_content))

    # Mỗi mặt bóc tách riêng (toạ độ mỗi ảnh khác nhau), mặt trước ưu tiên hơn
    ocr = extract_fields(back_lines)
    ocr.update(extract_fields(front_lines))

    # QR nằm ở mặt trước (CCCD 2021) hoặc mặt sau (Căn cước 2024)
    qr_text = decode_qr(front_img) or decode_qr(back_img)
    qr = parse_qr(qr_text) if qr_text else {}

    # Nếu người dùng gửi đảo 2 mặt thì MRZ nằm trong ảnh "mặt trước"
    mrz = parse_mrz(back_lines) or parse_mrz(front_lines)
    mrz_data = mrz["data"] if mrz else {}

    card_type = detect_card_type(front_lines + back_lines)
    sources = {"qr": qr, "ocr": ocr, "mrz": mrz_data}

    data, used = {}, {}
    for f in FIELDS:
        for src in PRIORITY.get(f, ("ocr", "qr", "mrz")):
            if sources[src].get(f):
                data[f] = sources[src][f]
                used[f] = src
                break
        else:
            data[f] = None

    data["issuing_authority"] = ISSUING_AUTHORITY.get(card_type)

    # Đối chiếu chéo giữa các nguồn
    warnings = []
    for f in ("id_number", "full_name", "date_of_birth", "sex", "date_of_expiry"):
        values = {src: sources[src][f] for src in ("qr", "ocr", "mrz") if sources[src].get(f)}
        if len({strip_accents(v).upper() for v in values.values()}) > 1:
            warnings.append(f"Trường '{f}' không khớp giữa các nguồn: {values}")
    if mrz_data.get("full_name_ascii") and data["full_name"] and not _same(mrz_data["full_name_ascii"], data["full_name"]):
        warnings.append(f"Họ tên không khớp với MRZ: '{data['full_name']}' vs '{mrz_data['full_name_ascii']}'")
    if mrz and not mrz["valid"]:
        warnings.append("MRZ đọc được nhưng sai checksum (ảnh mờ hoặc lóa)")
    _check_id_structure(data, warnings)

    missing = [f for f in ("id_number", "full_name", "date_of_birth") if not data[f]]
    if missing:
        warnings.append(f"Không đọc được các trường bắt buộc: {missing}")

    result = {
        "success": not missing,
        "card_type": card_type,
        "data": data,
        "sources": used,
        "qr_detected": bool(qr_text),
        "mrz": {"lines": mrz["lines"], "valid": mrz["valid"]} if mrz else None,
        "warnings": warnings,
        "message": "Đọc thông tin CCCD thành công" if not missing else "Không đọc đủ thông tin, vui lòng chụp lại ảnh rõ nét hơn",
    }
    if debug:
        result["debug"] = {
            "front_lines": [{"text": ln.text, "raw": ln.raw} for ln in front_lines],
            "back_lines": [{"text": ln.text, "raw": ln.raw} for ln in back_lines],
            "qr_raw": qr_text,
        }
    return result
