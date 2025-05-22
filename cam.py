import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)


model = YOLO('yolov8n.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

  
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            if cls == 0:  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    cv2.imshow('Human Detection Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
