
from ultralytics import YOLO
import cv2
model = YOLO("runs/detect/train14/weights/best.pt")

# Kamerayı aç (0 = default kamera, 1 ikinci kamera olur)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO ile tahmin yap
    results = model(frame)

    # Tahmin edilen kutuları ekrana çiz
    annotated_frame = results[0].plot()

    # Görüntüyü göster
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()