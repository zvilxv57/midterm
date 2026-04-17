import cv2
from ultralytics import YOLO

# 1. 初始化 YOLO 模型
model = YOLO('yolov8n.pt') 

# 2. 設定全域變數
target_id = None          # 目前被鎖定的目標 ID
current_boxes = []        # 儲存當前影格所有人的位置與 ID

# 3. 滑鼠點擊事件處理
def select_target(event, x, y, flags, param):
    global target_id, current_boxes
    if event == cv2.EVENT_LBUTTONDOWN:  # 當按下滑鼠左鍵
        for box in current_boxes:
            x1, y1, x2, y2, track_id = box
            # 判斷點擊位置是否落在某個框框內
            if x1 <= x <= x2 and y1 <= y <= y2:
                target_id = track_id
                print(f"--- 目標已更 換！目前鎖定 ID: {target_id} ---")
                break

# 4. 讀取影片
video_path = "midterm_video.mp4" # 請確保你的影片檔名正確
cap = cv2.VideoCapture(video_path)

# 建立視窗並綁定滑鼠事件
cv2.namedWindow("Concert_AI_Tracker")
cv2.setMouseCallback("Concert_AI_Tracker", select_target)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 執行追蹤 (persist=True 非常重要，它能讓 ID 在影格間維持一致)
    results = model.track(frame, persist=True, classes=0, verbose=False)

    # 準備更新當前影格的框資訊
    current_boxes = []

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, t_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            current_boxes.append((x1, y1, x2, y2, t_id))

            # 視覺化邏輯
            if t_id == target_id:
                # 鎖定目標：畫粗紅框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, f"TARGET ID: {t_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 計算中心點（用於未來控制馬達）
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            else:
                # 非目標：畫細綠框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f"ID: {t_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 顯示操作說明
    cv2.putText(frame, "Click on a person to LOCK", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Concert_AI_Tracker", frame)

    # 按 'q' 退出，按 'c' 清除鎖定
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        target_id = None
        print("已清除鎖定")

cap.release()
cv2.destroyAllWindows()


