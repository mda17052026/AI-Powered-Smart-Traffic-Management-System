from ultralytics import YOLO
import cv2

# ---------------- LOAD MODEL ----------------
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic.mp4")

# -------- Density Classification --------
def get_density(count):
    if count <= 4:
        return "LOW"
    elif count <= 9:
        return "MEDIUM"
    else:
        return "HIGH"

# -------- Convert density to priority --------
def density_score(d):
    if d == "LOW":
        return 1
    elif d == "MEDIUM":
        return 2
    else:
        return 3

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------- MODULE 1 ----------------
    resized = cv2.resize(frame, (640,640))
    blurred = cv2.GaussianBlur(resized, (5,5), 0)

    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(2.0,(8,8))
    cl = clahe.apply(l)

    enhanced = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

    # ---------------- MODULE 2 ----------------
    results = model(enhanced)
    annotated_frame = results[0].plot()

    # ---------------- MODULE 3 ----------------
    height, width, _ = enhanced.shape
    lane1_x = width // 3
    lane2_x = 2 * width // 3

    cv2.line(annotated_frame,(lane1_x,0),(lane1_x,height),(0,255,0),2)
    cv2.line(annotated_frame,(lane2_x,0),(lane2_x,height),(0,255,0),2)

    lane1_count = 0
    lane2_count = 0
    lane3_count = 0

    for box in results[0].boxes:
        x1,y1,x2,y2 = box.xyxy[0]
        center_x = int((x1+x2)/2)

        if center_x < lane1_x:
            lane1_count += 1
        elif center_x < lane2_x:
            lane2_count += 1
        else:
            lane3_count += 1

    # ---------------- MODULE 4 ----------------
    d1 = get_density(lane1_count)
    d2 = get_density(lane2_count)
    d3 = get_density(lane3_count)

    # ---------------- MODULE 5 (SIGNAL CONTROL) ----------------
    scores = {
        "Lane 1": density_score(d1),
        "Lane 2": density_score(d2),
        "Lane 3": density_score(d3)
    }

    # Sort lanes by congestion
    sorted_lanes = sorted(scores, key=scores.get, reverse=True)

    signals = {
        sorted_lanes[0]: "GREEN",
        sorted_lanes[1]: "YELLOW",
        sorted_lanes[2]: "RED"
    }

    # Display counts + signals
    cv2.putText(annotated_frame,
        f"Lane 1: {lane1_count} ({d1}) - {signals['Lane 1']}",
        (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.putText(annotated_frame,
        f"Lane 2: {lane2_count} ({d2}) - {signals['Lane 2']}",
        (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.putText(annotated_frame,
        f"Lane 3: {lane3_count} ({d3}) - {signals['Lane 3']}",
        (10,90), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("Module 5 - Adaptive Traffic Signal Control", annotated_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
