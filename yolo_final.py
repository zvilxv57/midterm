import cv2
import math
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# --- 1. 設定區 ---
VIDEO_PATH = "midterm_video.mp4" 
MODEL_SIZE = 'yolov8s.pt'
WINDOW_NAME = "Pro_Stable_Concert_Tracker"
ZOOM_W, ZOOM_H = 270, 405 # 特寫視窗尺寸

def init_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    kf.P *= 10.0
    kf.R = np.array([[2, 0], [0, 2]]) # 調小 R，讓特寫反應更靈敏
    kf.Q = np.eye(4) * 0.05          # 調小 Q，讓運動軌跡更平滑
    return kf

# --- 2. 狀態變數 ---
kf = init_kalman()
tracker_initialized = False
target_id = None
target_hist = None
lost_counter = 0
max_lost_frames = 90 # 提高到 3 秒，對抗長時遮擋
paused = False
current_frame_boxes = []

# --- 3. 顏色指紋：抗干擾優化 ---
def calculate_hist(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w, _ = frame.shape
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    roi_h, roi_w = y2 - y1, x2 - x1
    if roi_h < 20 or roi_w < 20: return None
    # 【工程修正】只取中心 40% 區域，徹底避開手臂擺動與背景
    roi = frame[y1+int(roi_h*0.3):y2-int(roi_h*0.3), x1+int(roi_w*0.3):x2-int(roi_w*0.3)]
    if roi.size == 0: return None
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_roi], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

# --- 4. 滑鼠點擊邏輯 ---
def select_target(event, x, y, flags, param):
    global target_id, target_hist, lost_counter, kf, tracker_initialized
    if event == cv2.EVENT_LBUTTONDOWN:
        for box in current_frame_boxes:
            x1, y1, x2, y2, t_id = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                tracker_initialized = False
                target_id = t_id
                target_hist = calculate_hist(param['frame'], [x1, y1, x2, y2])
                lost_counter = 0
                print(f">>> 重新校準鎖定 ID: {target_id}")
                break

# --- 5. 主程式 ---
model = YOLO(MODEL_SIZE)
cap = cv2.VideoCapture(VIDEO_PATH)

cv2.namedWindow(WINDOW_NAME)
params = {'frame': None}
cv2.setMouseCallback(WINDOW_NAME, select_target, params)

while cap.isOpened():
    if not paused:
        success, frame = cap.read()
        if not success: break
        params['frame'] = frame
        h_f, w_f, _ = frame.shape

        # YOLO 追蹤：降低 iou 門檻，防止重疊時框框合併
        results = model.track(frame, persist=True, classes=0, conf=0.3, iou=0.15, tracker="bytetrack.yaml", verbose=False)
        
        current_frame_boxes = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            for b, t in zip(boxes, track_ids):
                current_frame_boxes.append((*b, t))

        # 【物理預測】
        if tracker_initialized:
            kf.predict()
            p_cx, p_cy = kf.x[0][0], kf.x[1][0]
        else:
            p_cx, p_cy = (0, 0)

        best_candidate, found_now, max_score = None, False, -1

        # 【匹配邏輯強化】
        for x1, y1, x2, y2, t_id in current_frame_boxes:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            
            # 若 ID 依舊，但必須檢查距離是否合理 (防 ID 跳轉)
            if target_id is not None and t_id == target_id:
                dist_to_pred = math.sqrt((cx - p_cx)**2 + (cy - p_cy)**2) if tracker_initialized else 0
                if dist_to_pred < 200: # 距離預測點太遠就不信
                    best_candidate, found_now = (cx, cy, t_id), True
                    break
            
            # 若 ID 遺失，啟動顏色指紋 + 運動連續性
            if target_hist is not None and tracker_initialized:
                curr_hist = calculate_hist(frame, [x1, y1, x2, y2])
                if curr_hist is not None:
                    sim = 1 - cv2.compareHist(target_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
                    dist = math.sqrt((cx - p_cx)**2 + (cy - p_cy)**2)
                    # 【核心公式】調高座標連續性的權重 (0.4)，對抗變色燈光
                    score = sim * 0.6 + max(0, 1-(dist/400)) * 0.4
                    if score > 0.55 and score > max_score:
                        max_score, best_candidate = score, (cx, cy, t_id)

        # 【物理更新】
        if best_candidate:
            ncx, ncy, nid = best_candidate
            z = np.array([[ncx], [ncy]])
            if not tracker_initialized:
                kf.x = np.array([[z[0][0]], [z[1][0]], [0], [0]])
                tracker_initialized = True
            else:
                kf.update(z)
            target_id, lost_counter = nid, 0
        else:
            lost_counter += 1

    # --- 6. 畫面渲染 ---
    display_frame = frame.copy()

    if tracker_initialized and lost_counter < max_lost_frames:
        sm_cx, sm_cy = int(kf.x[0][0]), int(kf.x[1][0])
        
        # 特寫視窗
        x1_c, y1_c = max(0, sm_cx-ZOOM_W//2), max(0, sm_cy-ZOOM_H//2)
        x2_c, y2_c = min(w_f, sm_cx+ZOOM_W//2), min(h_f, sm_cy+ZOOM_H//2)
        
        zoom_view = frame[y1_c:y2_c, x1_c:x2_c].copy()
        if zoom_view.size > 0:
            zoom_view = cv2.resize(zoom_view, (ZOOM_W, ZOOM_H))
            b_c = (0, 0, 255) if lost_counter == 0 else (0, 255, 255)
            cv2.rectangle(zoom_view, (0,0), (ZOOM_W-1, ZOOM_H-1), b_c, 5)
            display_frame[h_f-ZOOM_H-20:h_f-20, w_f-ZOOM_W-20:w_f-20] = zoom_view
            label = "LOCKING" if lost_counter == 0 else f"PREDICTING ({lost_counter})"
            cv2.putText(display_frame, label, (w_f-ZOOM_W-20, h_f-ZOOM_H-30), 0, 0.7, b_c, 2)

    # 繪製所有路人框
    for x1, y1, x2, y2, t_id in current_frame_boxes:
        is_target = (t_id == target_id and lost_counter == 0)
        color = (0, 0, 255) if is_target else (0, 255, 0)
        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    cv2.putText(display_frame, "SPACE: Pause | C: Reset | Q: Quit", (20, 30), 0, 0.7, (255, 255, 255), 2)
    cv2.imshow(WINDOW_NAME, display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '): paused = not paused
    elif key == ord('c'):
        tracker_initialized = False
        target_id = None

cap.release()
cv2.destroyAllWindows()