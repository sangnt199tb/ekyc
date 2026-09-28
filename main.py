import cv2
from deepface import DeepFace

img1_path = "me.jpg"
img2_path = "selfie.jpg"

print("Đang phân tích khuôn mặt bằng model cao cấp (Facenet512)...")

try:
    # Sử dụng model_name='Facenet512' và detector_backend='retinaface'
    result = DeepFace.verify(
        img1_path=img1_path, 
        img2_path=img2_path, 
        model_name='Facenet512',
        detector_backend='retinaface'
    )
    
    print("\n--- KẾT QUẢ SO SÁNH CHÍNH XÁC ---")
    print(f"Cùng một người? -> {'ĐÚNG (MATCH)' if result['verified'] else 'SAI (NON MATCH)'}")
    print(f"Khoảng cách đo được (Distance): {result['distance']:.4f}")
    print(f"Ngưỡng cho phép (Threshold): {result['threshold']}")
    
    # Giải thích thêm cho đồ án:
    if result['distance'] < result['threshold']:
        print("👉 Nhận định: Hai khuôn mặt KHỚP NHAU.")
    else:
        print("👉 Nhận định: Hai khuôn mặt KHÁC NHAU hoàn toàn.")
        
except Exception as e:
    print(f"❌ Lỗi: {e}")