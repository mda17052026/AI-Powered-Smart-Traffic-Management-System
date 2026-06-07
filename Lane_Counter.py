from ultralytics import YOLO
import cv2

# ---------------- LOAD YOLO MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- OPEN VIDEO ----------------
cap = cv2.VideoCapture("traffic.mp4")

if not cap.isOpened():
    print("Error opening video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------- MODULE 1 (PREPROCESSING) ----------------
    resized = cv2.resize(frame, (640, 640))

    # Noise reduction
    blurred = cv2.GaussianBlur(resized, (5,5), 0)

    # CLAHE Contrast Enhancement
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # ---------------- MODULE 2 (YOLO DETECTION) ----------------
    results = model(enhanced)
    annotated_frame = results[0].plot()

    # ---------------- MODULE 3 (LANE DETECTION & COUNTING) ----------------
    height, width, _ = enhanced.shape

    lane1_x = width // 3
    lane2_x = 2 * width // 3

    # Draw lane lines
    cv2.line(annotated_frame, (lane1_x, 0), (lane1_x, height), (0,255,0), 2)
    cv2.line(annotated_frame, (lane2_x, 0), (lane2_x, height), (0,255,0), 2)

    # Initialize counters
    lane1_count = 0
    lane2_count = 0
    lane3_count = 0

    # Count vehicles per lane
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]

        center_x = int((x1 + x2) / 2)

        if center_x < lane1_x:
            lane1_count += 1
        elif center_x < lane2_x:
            lane2_count += 1
        else:
            lane3_count += 1

    # Display counts
    cv2.putText(annotated_frame, f"Lane 1: {lane1_count}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(annotated_frame, f"Lane 2: {lane2_count}", (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.putText(annotated_frame, f"Lane 3: {lane3_count}", (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # ---------------- SHOW OUTPUT ----------------
    cv2.imshow("Module 3 - Lane Detection & Vehicle Counting", annotated_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
