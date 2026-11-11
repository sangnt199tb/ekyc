from fastapi import FastAPI, UploadFile, File, HTTPException
from deepface import DeepFace
import os
import threading
import uuid
from cccd_ocr import read_cccd

app = FastAPI(title="eKYC Face Verification API", version="1.1")

# Thư mục lưu tạm ảnh nhận được từ Java gửi sang
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# DeepFace dùng chung một model TensorFlow trong tiến trình: chỉ cho một request so khớp chạy tại một thời điểm,
# tránh hai request cùng nạp, cùng dùng model. Các request khác chờ trong threadpool, không chặn server
# (OCR vẫn chạy song song).
_face_lock = threading.Lock()


def _save_temp(upload: UploadFile, prefix: str) -> str:
    """
    Lưu ảnh vào file tạm có tên ngẫu nhiên. Không dùng tên file client gửi: hai request trùng tên sẽ ghi đè ảnh của
    nhau (so khớp nhầm người), và tên chứa "../" có thể ghi ra ngoài TEMP_DIR.
    """
    path = os.path.join(TEMP_DIR, f"{prefix}_{uuid.uuid4().hex}.jpg")
    with open(path, "wb") as buffer:
        buffer.write(upload.file.read())
    return path


def _remove_quietly(path):
    if path and os.path.exists(path):
        os.remove(path)


@app.post("/api/verify-face")
def verify_face(cccd_image: UploadFile = File(...), selfie_image: UploadFile = File(...)):
    """
    API nhận 2 file ảnh (cccd_image và selfie_image) từ Spring Boot,
    sau đó thực hiện so sánh khuôn mặt và trả về kết quả JSON.
    Khai báo `def` (không async): DeepFace chạy đồng bộ và lâu (vài giây), để trong `async def` sẽ chặn event loop,
    cả server không nhận request nào khác. Với `def`, FastAPI chạy hàm trong threadpool.
    """
    cccd_path = None
    selfie_path = None
    try:
        # 1. Lưu tạm 2 file ảnh nhận được từ request
        cccd_path = _save_temp(cccd_image, "cccd")
        selfie_path = _save_temp(selfie_image, "selfie")

        # 2. Gọi DeepFace để so sánh (dùng Facenet512 và retinaface như đã tối ưu)
        with _face_lock:
            result = DeepFace.verify(
                img1_path=cccd_path,
                img2_path=selfie_path,
                model_name='Facenet512',
                detector_backend='retinaface'
            )

        # 3. Trả kết quả về cho Spring Boot dưới dạng JSON
        return {
            "success": True,
            "verified": bool(result["verified"]),
            "distance": float(result["distance"]),
            "threshold": float(result["threshold"]),
            "message": "Khuôn mặt khớp nhau" if result["verified"] else "Khuôn mặt không khớp"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Không thể phát hiện khuôn mặt hoặc lỗi xử lý ảnh."
        }
    finally:
        # 4. Luôn dọn file tạm, kể cả khi lỗi
        _remove_quietly(cccd_path)
        _remove_quietly(selfie_path)


@app.post("/api/ocr-cccd")
def ocr_cccd(front_image: UploadFile = File(...), back_image: UploadFile = File(...), debug: bool = False):
    """
    API nhận ảnh 2 mặt CCCD / thẻ Căn cước (front_image, back_image) và trả về toàn bộ thông tin trên thẻ.
    Hỗ trợ CCCD gắn chip (2021) và thẻ Căn cước (2024). Khai báo `def` (không async) để FastAPI chạy
    trong threadpool, tránh chặn event loop khi OCR.
    """
    try:
        return read_cccd(front_image.file.read(), back_image.file.read(), debug=debug)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Lỗi xử lý ảnh CCCD."
        }

if __name__ == "__main__":
    import uvicorn
    # Chạy server ở cổng 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
