from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Label jari tangan
finger_names = ["Ibu Jari", "Telunjuk", "Tengah", "Manis", "Kelingking"]
finger_indices = [4, 8, 12, 16, 20]  # Indeks landmark untuk setiap jari

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

    # Konversi warna untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(rgb_frame)

    # Deteksi pose dengan YOLOv8
    results = model.predict(frame, imgsz=640, conf=0.5)

    for result in results:
        # Perkecil ukuran bounding box dengan faktor tertentu
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            shrink_factor = 0.8  # Faktor pengecilan
            width = x2 - x1
            height = y2 - y1
            x1 += int(width * (1 - shrink_factor) / 2)
            x2 -= int(width * (1 - shrink_factor) / 2)
            y1 += int(height * (1 - shrink_factor) / 2)
            y2 -= int(height * (1 - shrink_factor) / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        annotated_frame = result.plot()

    # Deteksi tangan dengan MediaPipe
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))
            
            # Tandai setiap jari dengan label yang benar
            for i, index in enumerate(finger_indices):
                landmark = hand_landmarks.landmark[index]
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.putText(frame, finger_names[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow("YOLOv8 Pose Estimation & MediaPipe Hands", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
