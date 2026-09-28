"""
Đọc mã QR trên CCCD gắn chip / thẻ Căn cước.
Nội dung QR có dạng: số định danh|số CMND cũ|họ tên|ngày sinh(ddmmyyyy)|giới tính|địa chỉ|ngày cấp(ddmmyyyy)
"""
import re

import cv2
import numpy as np
import zxingcpp


def decode_qr(img: np.ndarray) -> str | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # QR trên thẻ khá nhỏ, thử thêm ảnh phóng to nếu ảnh gốc không đọc được
    candidates = [gray]
    if max(gray.shape) < 2500:
        candidates.append(cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))

    for candidate in candidates:
        for barcode in zxingcpp.read_barcodes(candidate, formats=zxingcpp.BarcodeFormat.QRCode):
            if barcode.valid and "|" in barcode.text:
                return barcode.text
    return None


def _date(value: str) -> str | None:
    value = value.strip()
    if re.fullmatch(r"\d{8}", value):
        return f"{value[:2]}/{value[2:4]}/{value[4:]}"
    return None


def parse_qr(text: str) -> dict:
    parts = [p.strip() for p in text.split("|")]
    parts += [""] * (7 - len(parts))

    data = {
        "id_number": parts[0] if re.fullmatch(r"\d{12}", parts[0]) else None,
        "old_id_number": parts[1] if re.fullmatch(r"\d{9}|\d{12}", parts[1]) else None,
        "full_name": parts[2].upper() or None,
        "date_of_birth": _date(parts[3]),
        "sex": parts[4] if parts[4] in ("Nam", "Nữ") else None,
        "place_of_residence": parts[5] or None,
        "date_of_issue": _date(parts[6]),
    }
    return {k: v for k, v in data.items() if v}
