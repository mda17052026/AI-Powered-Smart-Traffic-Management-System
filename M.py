import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("traffic.mp4")

vehicle_classes = ["car","bus","truck","motorcycle"]

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame,(900,600))

    height, width, _ = frame.shape

    lane1 = width//3
    lane2 = (width//3)*2

    # dashboards
    video_input = frame.copy()
    detection_frame = frame.copy()
    lane_frame = frame.copy()

    lane1_count = 0
    lane2_count = 0
    lane3_count = 0

    # YOLO detection
    results = model(frame)

    for r in results:

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):

            label = model.names[int(cls)]

            if label in vehicle_classes:

                x1,y1,x2,y2 = map(int,box)

                cx = (x1+x2)//2

                # vehicle detection dashboard
                cv2.rectangle(detection_frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(detection_frame,label,(x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                # lane counting
                if cx < lane1:
                    lane1_count += 1
                elif cx < lane2:
                    lane2_count += 1
                else:
                    lane3_count += 1

    # lane detection dashboard
    cv2.line(lane_frame,(lane1,0),(lane1,height),(255,255,0),3)
    cv2.line(lane_frame,(lane2,0),(lane2,height),(255,255,0),3)

    cv2.putText(lane_frame,f"Lane1: {lane1_count}",(50,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    cv2.putText(lane_frame,f"Lane2: {lane2_count}",(350,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    cv2.putText(lane_frame,f"Lane3: {lane3_count}",(650,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    # traffic density dashboard
    density_frame = np.zeros((300,500,3),dtype=np.uint8)

    cv2.putText(density_frame,f"Lane1 Vehicles: {lane1_count}",(50,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.putText(density_frame,f"Lane2 Vehicles: {lane2_count}",(50,130),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    cv2.putText(density_frame,f"Lane3 Vehicles: {lane3_count}",(50,190),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

    # signal decision
    counts = [lane1_count,lane2_count,lane3_count]
    max_lane = counts.index(max(counts))

    signals = ["RED","RED","RED"]
    signals[max_lane] = "GREEN"

    signal_frame = np.zeros((300,500,3),dtype=np.uint8)

    cv2.putText(signal_frame,f"Lane1 Signal: {signals[0]}",(50,70),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(signal_frame,f"Lane2 Signal: {signals[1]}",(50,130),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    cv2.putText(signal_frame,f"Lane3 Signal: {signals[2]}",(50,190),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    # show dashboards
    cv2.imshow("1. Video Input",video_input)
    cv2.imshow("2. Vehicle Detection",detection_frame)
    cv2.imshow("3. Lane Detection & Counting",lane_frame)
    cv2.imshow("4. Traffic Density",density_frame)
    cv2.imshow("5. Signal Decision",signal_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()