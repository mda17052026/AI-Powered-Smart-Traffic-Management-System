import cv2

# Load traffic video
cap = cv2.VideoCapture("traffic.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame
    resized = cv2.resize(frame, (640, 640))

    # Reduce noise
    blurred = cv2.GaussianBlur(resized, (5,5), 0)

    # Improve contrast
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    enhanced = cv2.merge((cl,a,b))
    final = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Display output
    cv2.imshow("Preprocessed Video", final)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()