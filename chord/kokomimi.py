from ultralytics import YOLO
import cv2

# YOLOv8n モデルの読み込み（初回実行時に自動ダウンロード）
model = YOLO("yolov8n.pt")

# カメラを開く（0 = PCの内蔵カメラ）
cap = cv2.VideoCapture(0)  # or 動画ファイル: cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 人物検出（classes=0はpersonクラス）
    results = model(frame, classes=[0])


    # 検出された人物のバウンディングボックスを描画
    annotated_frame = results[0].plot()

    # 検出された人数をカウント
    num_people = len(results[0].boxes)
    cv2.putText(annotated_frame, f"People: {num_people}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 映像表示
    cv2.imshow("People Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
