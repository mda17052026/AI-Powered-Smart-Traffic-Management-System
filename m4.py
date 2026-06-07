from ultralytics import YOLO
import cv2

model=YOLO("yolov8n.pt")
cap=cv2.VideoCapture("traffic.mp4")

while True:
    ret,frame=cap.read()
    if not ret:
        break

    frame=cv2.resize(frame,(640,640))
    results=model(frame)
    output=results[0].plot()

    h,w,_=frame.shape
    l1=w//3
    l2=2*w//3

    c1=c2=c3=0

    for box in results[0].boxes:
        x1,_,x2,_=box.xyxy[0]
        cx=int((x1+x2)/2)

        if cx<l1:
            c1+=1
        elif cx<l2:
            c2+=1
        else:
            c3+=1

    def d(c):
        return "LOW" if c<=4 else "MEDIUM" if c<=9 else "HIGH"

    cv2.putText(output,f"Lane1:{d(c1)}",(10,30),0,0.7,(0,255,0),2)
    cv2.putText(output,f"Lane2:{d(c2)}",(10,60),0,0.7,(0,255,0),2)
    cv2.putText(output,f"Lane3:{d(c3)}",(10,90),0,0.7,(0,255,0),2)

    cv2.imshow("Traffic Density",output)

    if cv2.waitKey(25)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()