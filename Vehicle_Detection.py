from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------- MODULE 1 ----------------
    # Resize frame
    resized = cv2.resize(frame, (640, 640))

    # Noise reduction (Gaussian Blur)
    blurred = cv2.GaussianBlur(resized, (5,5), 0)

    # ----- CLAHE Contrast Enhancement -----
    # Convert to LAB color space
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE on Lightness channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # Merge channels back
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Show preprocessing result (Module 1 output)
    cv2.imshow("Preprocessed Frame (CLAHE)", enhanced)

    # ---------------- MODULE 2 ----------------
    # Vehicle detection using enhanced frame
    results = model(enhanced)

    annotated_frame = results[0].plot()

    # Show detection result
    cv2.imshow("Vehicle Detection", annotated_frame)

    # Exit key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
