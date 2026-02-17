from ultralytics import YOLO
import cv2
import cvzone
# import funcations_

# Load YOLOv8 face model PROPERLY
model__ = YOLO(r'C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\Scratch, YOLO - Face Detection\Scratch\yolov8n-face-lindevs.onnx')
facemodel = YOLO(model=model__)
#this is for web_cam video (line 9)
cap = cv2.VideoCapture(0)
#we can do "cap = cv2.VideoCapture("rtsp://...")" for RTSP stream (line 9)
#example :rtsp://admin:admin123@192.168.1.108:554/Streaming/Channels/101
        #:rtsp://user:pass@10.0.0.25:8554/live

while cap.isOpened():
    ret, video = cap.read()
    if not ret:
        break

    video = cv2.resize(video, (900, 700)) #this also return speed

    results = facemodel.predict(video, conf=0.4, verbose=False)

    #we can use if condition to check is user want to give input of image, video or stream (line 19)
    for r in results: #this can be used with condition for image, video and stream
        if r.boxes is None: #this returns none if no face detected
            continue
        for box in r.boxes: #detected bounding boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0]) #here if 0 returns the coordinates it means 1 will return confidence and 2 will return class ID
            w, h = x2 - x1, y2 - y1 
            cvzone.cornerRect(video, (x1, y1, w, h), l=9, rt=3)

    cv2.imshow("frame", video)
    if cv2.waitKey(1) & 0xFF == 27: #27 is to quit stream using esc key # 1 is to wait for 1 millisecond
        break

cap.release() #Free webcam
cv2.destroyAllWindows() #Close OpenCV windows