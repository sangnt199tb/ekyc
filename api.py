from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from deepface import DeepFace
import os
from cccd_ocr import read_cccd

app = FastAPI(title="eKYC Face Verification API", version="1.0")

# Thư mục lưu tạm ảnh nhận được từ Java gửi sang
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/api/verify-face")
async def verify_face(cccd_image: UploadFile = File(...), selfie_image: UploadFile = File(...)):
    """
    API nhận 2 file ảnh (cccd_image và selfie_image) từ Spring Boot,
    sau đó thực hiện so sánh khuôn mặt và trả về kết quả JSON.
    """
    try:
        # 1. Lưu tạm 2 file ảnh nhận được từ request
        cccd_path = os.path.join(TEMP_DIR, cccd_image.filename)
        selfie_path = os.path.join(TEMP_DIR, selfie_image.filename)
        
        with open(cccd_path, "wb") as buffer:
            buffer.write(await cccd_image.read())
            
        with open(selfie_path, "wb") as buffer:
            buffer.write(await selfie_image.read())
            
        # 2. Gọi DeepFace để so sánh (dùng Facenet512 và retinaface như đã tối ưu)
        result = DeepFace.verify(
            img1_path=cccd_path,
            img2_path=selfie_path,
            model_name='Facenet512',
            detector_backend='retinaface'
        )
        
        # 3. Dọn dẹp file tạm sau khi xử lý xong
        if os.path.exists(cccd_path): os.remove(cccd_path)
        if os.path.exists(selfie_path): os.remove(selfie_path)
        
        # 4. Trả kết quả về cho Spring Boot dưới dạng JSON
        return {
            "success": True,
            "verified": bool(result["verified"]),
            "distance": float(result["distance"]),
            "threshold": float(result["threshold"]),
            "message": "Khuôn mặt khớp nhau" if result["verified"] else "Khuôn mặt không khớp"
        }
        
    except Exception as e:
        # Xóa file tạm nếu có lỗi xảy ra
        if os.path.exists(cccd_path): os.remove(cccd_path)
        if os.path.exists(selfie_path): os.remove(selfie_path)
        
        return {
            "success": False,
            "error": str(e),
            "message": "Không thể phát hiện khuôn mặt hoặc lỗi xử lý ảnh."
        }

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