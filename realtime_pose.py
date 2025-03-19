from ultralytics import YOLO
import cv2

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Gagal membuka kamera!")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame!")
        break

    # Deteksi pose (Convert frame ke RGB karena OpenCV default BGR)
    results = model.predict(frame, imgsz=640)  # Pastikan input format benar

    # Tampilkan hasil
    for result in results:
        annotated_frame = result.plot()  # Tambahkan anotasi
        cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
