from ultralytics import YOLO
import cv2
import cvzone
import funcations_

# Load YOLOv8 face model PROPERLY
facemodel = YOLO(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\Scratch - ANN\yolov8n-face-lindevs.onnx")

#this is for web_cam video (line 9)
# mode = input('enter face detection source input (image, video, web_cam, rtps_stream)')
# path = input('enter path to source input')
# if mode == 'image':
#     frame , img = funcations_.image_detect(path=path)

# elif mode == 'video':
#     results, frame = funcations_.video_detect(path=path)

# elif mode == 'web_cam':
#     results, frame = funcations_.webcam_detect(path=path)

# else:
#     results, frame = funcations_.rtsp_detect(path=path)

#funcation called from funcations_.py folder.
#this is to get input from user about input souces
#args: path
cap = funcations_.cap()
#we can do "cap = cv2.VideoCapture("rtsp://...")" for RTSP stream (line 9)
#example :rtsp://admin:admin123@192.168.1.108:554/Streaming/Channels/101
        #:rtsp://user:pass@10.0.0.25:8554/live

while cap.isOpened():
    ret, video = cap.read()
#     if not ret:
#         break

#     frame_count += 1
#     if frame_count % 2 != 0:
#         continue   # Skip every alternate frame  

#funcation called from funcations_.py folder.
#this is to drop every alternate frames from video/stream input
#this to control the speed of frames/model
    funcations_.skip_frame(ret=ret)

    video = cv2.resize(video, (900, 700)) #this also return speed

    results = facemodel.predict(video, conf=0.4, verbose=False)

    #we can use if condition to check is user want to give input of image, video or stream (line 19)
    for r in results: #this can be used with condition for image, video and stream
        if r.boxes is None: #this returns none if no face detected
            continue
        for box in r.boxes: #detected bounding boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0]) #here if 0 returns the coordinates it means 1 will return confidence and 2 will return class ID
            conf = float(box.conf[0])*100
            class_id = int(box.cls[0])
            class_name = r.names[class_id]
            confidence = round(float(box.conf[0]), 2)
            label = f"{class_name} {confidence:.2f}"    
            w, h = x2 - x1, y2 - y1 
            cvzone.cornerRect(video, (x1, y1, w, h), l=9, rt=3)

    cv2.imshow("frame", video)
#funcation called from funcations_.py
#this to control display around the box
    cv2.putText(funcations_.display_controls(video=video, conf=confidence, x1=x1, y1=y1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    if cv2.waitKey(1) & 0xFF == 27: #27 is to quit stream using esc key # 1 is to wait for 1 millisecond
        break

cap.release() #Free webcam
cv2.destroyAllWindows() #Close OpenCV windows

