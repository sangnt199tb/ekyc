"""
Bóc tách các trường thông tin từ các dòng OCR dựa trên nhãn (label) và vị trí.

Ý tưởng:
1. Tìm các dòng chứa nhãn ("Họ và tên / Full name", "Ngày sinh / Date of birth"...).
   Nhãn được so khớp trên cả bản tiếng Việt (bỏ dấu) và tiếng Anh nên chịu được lỗi OCR.
2. Giá trị của một trường = phần chữ nằm sau nhãn trên cùng dòng, hoặc các dòng nằm ngay bên dưới
   nhãn và bắt đầu trong phạm vi chiều ngang của nhãn (để phân biệt 2 cột, vd "Ngày sinh" và "Giới tính").
"""
import re
import unicodedata
from dataclasses import dataclass, field

from .engine import TextLine, looks_like_mrz

# Thứ tự quan trọng: nhãn cụ thể hơn đặt trước
LABELS: list[tuple[str, list[str]]] = [
    ("id_number", [r"so dinh danh", r"identification number", r"\bso\s*/\s*no\b"]),
    ("full_name", [r"ho,? chu dem va ten", r"ho va ten", r"full name", r"ten khai sinh"]),
    ("place_of_birth", [r"noi dang ky khai sinh", r"dang ky khai sinh", r"place of birth"]),
    ("date_of_birth", [r"ngay,? thang,? nam sinh", r"ngay sinh", r"date of birth"]),
    ("sex", [r"gioi tinh", r"\bsex\b"]),
    ("nationality", [r"quoc t[iy]ch", r"nationality"]),
    ("place_of_origin", [r"que quan", r"place of origin"]),
    ("place_of_residence", [r"noi thuong tru", r"noi cu tru", r"place of residence"]),
    ("date_of_expiry", [r"co gia tri den", r"het han", r"date of expiry"]),
    ("date_of_issue", [r"ngay,? thang,? nam cap", r"date of issue", r"ngay,? thang,? nam(?! ?(sinh|cap|het))", r"date,? month,? year"]),
    ("personal_identification", [r"dac diem nhan dang", r"personal identification"]),
]

# Các dòng không bao giờ là giá trị của trường nào
NOISE = re.compile(
    r"cong hoa|doc lap|socialist|independence|can cuoc|identity card|citizen|"
    r"bo cong an|ministry|public security|cuc truong|director|cuc canh sat|police|"
    r"ngon tro|index finger|fingerprint"
)

DATE_RE = re.compile(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})")
SEPARATORS = " :;/.,-_|"


def strip_accents(text: str) -> str:
    """Bỏ dấu từng ký tự, giữ nguyên độ dài chuỗi để map vị trí về chuỗi gốc."""
    out = []
    for c in text:
        if c in "đĐ":
            out.append("d")
        else:
            out.append(unicodedata.normalize("NFD", c)[0].lower())
    return "".join(out)


@dataclass
class Anchor:
    """Vị trí một nhãn trên ảnh."""
    field: str
    x0: float
    x1: float
    y0: float
    y1: float
    inline: str                       # Phần giá trị nằm cùng dòng, sau nhãn
    raw: str                          # Toàn bộ dòng theo RapidOCR (đọc số chính xác hơn)
    values: list[TextLine] = field(default_factory=list)

    @property
    def h(self) -> float:
        return self.y1 - self.y0


def _find_labels(line: TextLine) -> list[Anchor]:
    """Tìm các nhãn trong 1 dòng. Một dòng có thể chứa 2 nhãn, vd 'Giới tính / Sex: Nam  Quốc tịch: Việt Nam'."""
    hits = []  # (start, end, field) trên chuỗi VietOCR
    for source in (line.text, line.raw):
        norm = strip_accents(source)
        scale = len(line.text) / max(len(source), 1)
        for fname, patterns in LABELS:
            for p in patterns:
                for m in re.finditer(p, norm):
                    start, end = int(m.start() * scale), int(m.end() * scale)
                    if any(start < e and s < end for s, e, _ in hits):
                        continue
                    hits.append((start, end, fname))
        if hits:
            break  # Ưu tiên nhãn tìm thấy trên VietOCR
    if not hits:
        return []

    hits.sort()
    # Gộp các nhãn liền nhau của cùng một trường (phần tiếng Việt + phần tiếng Anh)
    groups: list[list] = []
    for s, e, fname in hits:
        if groups and groups[-1][2] == fname:
            groups[-1][1] = max(groups[-1][1], e)
        else:
            groups.append([s, e, fname])

    anchors = []
    width = line.x1 - line.x0
    n = max(len(line.text), 1)
    for i, (s, e, fname) in enumerate(groups):
        seg_end = groups[i + 1][0] if i + 1 < len(groups) else len(line.text)
        inline = line.text[e:seg_end].strip(SEPARATORS)
        # Bỏ phần nhãn tiếng Anh còn sót sau dấu '/'
        inline = re.sub(r"^[A-Za-z ,]+:\s*", "", inline) if ":" in inline else inline
        # Phần chữ đứng trước nhãn đầu tiên cũng thuộc nhãn, vd "Ngày, tháng, năm" trong "Ngày, tháng, năm hết hạn"
        start = 0 if i == 0 else s
        anchors.append(Anchor(
            field=fname,
            x0=line.x0 + width * start / n,
            x1=line.x0 + width * seg_end / n,
            y0=line.y0, y1=line.y1,
            inline=inline.strip(SEPARATORS),
            raw=line.raw,
        ))
    return anchors


def _assign_values(anchors: list[Anchor], others: list[TextLine]) -> None:
    """Gán mỗi dòng giá trị cho nhãn gần nhất phía trên (hoặc bên trái cùng hàng)."""
    for ln in others:
        best, best_dist = None, None
        for a in anchors:
            tol = 1.5 * a.h
            same_row = abs(ln.yc - (a.y0 + a.y1) / 2) < 0.5 * a.h and ln.x0 >= a.x1 - tol
            below = ln.y0 >= a.y0 + 0.5 * a.h and a.x0 - tol <= ln.x0 <= a.x1
            if not (same_row or below):
                continue
            dist = 0 if same_row else ln.y0 - a.y1
            if best_dist is None or dist < best_dist:
                best, best_dist = a, dist
        if best is not None:
            best.values.append(ln)

    # Chỉ giữ các dòng nối tiếp nhau (khoảng cách nhỏ), tránh lấy nhầm chữ ở xa
    for a in anchors:
        kept, last_y1 = [], a.y1
        for ln in sorted(a.values, key=lambda v: v.y0):
            if ln.y0 - last_y1 <= 1.6 * max(ln.h, a.h):
                kept.append(ln)
                last_y1 = max(last_y1, ln.y1)
        a.values = kept


def _fmt_date(text: str) -> str | None:
    m = DATE_RE.search(text)
    if not m:
        return None
    d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if 1 <= d <= 31 and 1 <= mth <= 12 and 1900 <= y <= 2200:
        return f"{d:02d}/{mth:02d}/{y}"
    return None


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(SEPARATORS)
    return re.sub(r"\s*,\s*", ", ", text)


def _field_value(fname: str, a: Anchor) -> str | None:
    parts = ([a.inline] if a.inline else []) + [ln.text for ln in a.values]
    raw_parts = ([a.inline] if a.inline else []) + [ln.raw for ln in a.values]
    if not parts:
        return None

    if fname.startswith("date_"):
        joined = " ".join([a.raw] + raw_parts + parts)
        if "khong thoi han" in strip_accents(joined):
            return "Không thời hạn"
        return _fmt_date(joined)

    if fname == "sex":
        norm = strip_accents(" ".join(parts))
        if re.search(r"\bnu\b", norm):
            return "Nữ"
        if re.search(r"\bnam\b", norm):
            return "Nam"
        return None

    if fname == "nationality":
        norm = strip_accents(" ".join(parts))
        return "Việt Nam" if "nam" in norm else _clean_text(parts[0])

    if fname == "id_number":
        m = re.search(r"\d{12}", " ".join([a.raw] + raw_parts + parts).replace(" ", ""))
        return m.group(0) if m else None

    if fname == "full_name":
        # Họ tên chỉ nằm trên 1-2 dòng, bỏ ký tự lạ
        name = " ".join(parts[:2])
        name = re.sub(r"[^\w\s]", " ", name).replace("_", " ")
        name = re.sub(r"\d", "", name)
        return re.sub(r"\s+", " ", name).strip().upper() or None

    # Các trường địa chỉ / mô tả: nối nhiều dòng
    return _clean_text(", ".join(p.strip(SEPARATORS) for p in parts if p.strip(SEPARATORS))) or None


def detect_card_type(lines: list[TextLine]) -> str:
    text = strip_accents(" ".join(ln.text + " " + ln.raw for ln in lines))
    new_score = sum(k in text for k in ("dinh danh", "khai sinh", "noi cu tru", "bo cong an", "public security"))
    old_score = sum(k in text for k in ("cong dan", "citizen", "que quan", "thuong tru", "co gia tri den", "cuc truong"))
    if new_score > old_score:
        return "CAN_CUOC_2024"
    if old_score > new_score:
        return "CCCD_CHIP_2021"
    return "UNKNOWN"


def count_labels(lines: list[TextLine]) -> int:
    return sum(len(_find_labels(ln)) for ln in lines)


def extract_fields(lines: list[TextLine]) -> dict:
    anchors: list[Anchor] = []
    others: list[TextLine] = []
    for ln in lines:
        found = _find_labels(ln)
        if found:
            anchors.extend(found)
        elif not looks_like_mrz(ln.raw) and not NOISE.search(strip_accents(ln.text)):
            others.append(ln)

    _assign_values(anchors, others)

    data: dict = {}
    for a in anchors:
        if a.field in data:
            continue
        value = _field_value(a.field, a)
        if value:
            data[a.field] = value

    # Số định danh: số 12 chữ số, không phụ thuộc nhãn
    if "id_number" not in data:
        for ln in lines:
            if looks_like_mrz(ln.raw):
                continue
            m = re.search(r"(?<!\d)\d{12}(?!\d)", ln.raw.replace(" ", ""))
            if m:
                data["id_number"] = m.group(0)
                break
    return data
