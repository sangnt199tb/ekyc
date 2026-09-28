"""
Đọc dải MRZ (chuẩn ICAO 9303 - TD1, 3 dòng x 30 ký tự) ở mặt sau thẻ.
Ví dụ:
    IDVNM0940002981034094000298<<2
    9412226M3412226VNM<<<<<<<<<<<8
    NGUYEN<<TRONG<SANG<<<<<<<<<<<<
"""
import re
from datetime import date

from .engine import TextLine, looks_like_mrz

# Ký tự OCR hay đọc nhầm với chữ số
_TO_DIGIT = str.maketrans("OQDIlLZSBG", "0001112586")
_D = "[0-9OQDIlLZSBG]"

_LINE1 = re.compile(rf"ID.?VNM({_D}{{9}})({_D})({_D}{{12}})")
_LINE2 = re.compile(rf"({_D}{{6}})({_D})([MF<])({_D}{{6}})({_D})([A-Z]{{3}})")


def _check_digit(data: str) -> str:
    total = 0
    for i, c in enumerate(data):
        if c.isdigit():
            v = int(c)
        elif c.isalpha():
            v = ord(c) - ord("A") + 10
        else:
            v = 0
        total += v * (7, 3, 1)[i % 3]
    return str(total % 10)


def _yymmdd(value: str, is_birth: bool) -> str | None:
    yy, mm, dd = int(value[:2]), int(value[2:4]), int(value[4:])
    if is_birth:
        year = 2000 + yy if yy <= date.today().year % 100 else 1900 + yy
    else:
        year = 2000 + yy
    try:
        return date(year, mm, dd).strftime("%d/%m/%Y")
    except ValueError:
        return None


def _clean(raw: str) -> str:
    s = raw.upper().replace(" ", "").replace("«", "<<").replace("‹", "<")
    return re.sub(r"[^A-Z0-9<]", "<", s)


def parse_mrz(lines: list[TextLine]) -> dict | None:
    mrz_lines = [_clean(ln.raw) for ln in lines if looks_like_mrz(ln.raw)]
    if not mrz_lines:
        return None

    data, checks = {}, []
    line1 = line2 = line3 = None
    for s in mrz_lines:
        if line1 is None and (m := _LINE1.search(s)):
            line1 = s
            doc_no, doc_check, full_id = (g.translate(_TO_DIGIT) for g in m.groups())
            checks.append(_check_digit(doc_no) == doc_check)
            data["id_number"] = full_id
        elif line2 is None and (m := _LINE2.search(s)):
            line2 = s
            dob, dob_check, sex, expiry, exp_check, nationality = m.groups()
            dob, dob_check = dob.translate(_TO_DIGIT), dob_check.translate(_TO_DIGIT)
            expiry, exp_check = expiry.translate(_TO_DIGIT), exp_check.translate(_TO_DIGIT)
            checks.append(_check_digit(dob) == dob_check)
            checks.append(_check_digit(expiry) == exp_check)
            data["date_of_birth"] = _yymmdd(dob, is_birth=True)
            data["date_of_expiry"] = _yymmdd(expiry, is_birth=False)
            data["sex"] = {"M": "Nam", "F": "Nữ"}.get(sex)
            data["nationality"] = "Việt Nam" if nationality == "VNM" else nationality
        elif line3 is None and re.fullmatch(r"[A-Z<]{10,}", s) and "<" in s:
            line3 = s
            parts = [p.replace("<", " ").strip() for p in re.split(r"<<+", s.strip("<"))]
            data["full_name_ascii"] = " ".join(p for p in parts if p)

    return {
        "lines": [s for s in (line1, line2, line3) if s],
        "valid": bool(checks) and all(checks),
        "data": {k: v for k, v in data.items() if v},
    }
