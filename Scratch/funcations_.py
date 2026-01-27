from ultralytics import YOLO
import cv2
import cvzone

facemodel = YOLO(r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\Scratch - ANN\yolov8n-face-lindevs.onnx")

def image_detect(path):
    img = cv2.imread(path)
    results = facemodel.predict(img)
    return results, img

def video_detect(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = facemodel.predict(frame)
        yield results, frame

def webcam_detect():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = facemodel.predict(frame)
        yield results, frame

def rtsp_detect(rtsp_url):
    cap = cap = cv2.VideoCapture(rtsp_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = facemodel.predict(frame)
        yield results, frame 

class cap:
    def __init__(self):
        pass

    def select_input_mode(self):
        mode = input('enter face detection source input (image, video, web_cam, rtps_stream)')
        path = input('enter path to source input')
        if mode == 'image':
            frame , img = self.image_detect(path=path)

        elif mode == 'video':
            results, frame = self.video_detect(path=path)

        elif mode == 'web_cam':
            results, frame = self.webcam_detect(path=path)

        else:
            results, frame = self.rtsp_detect(path=path)

        return results, frame, img

def skip_frame(ret):
    while True:
        if not ret:
            break
        frame_count += 1
        if frame_count % 2 != 0:
            continue   # Skip every alternate frame

    return 

def display_controls(video, label, conf, x1, y1):
    display = cv2.putText(
    video,
    label,
    f"{conf}",
    (x1, y1 - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2
)
    return display