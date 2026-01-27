from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")   #  trainable base model

    model.train(
        data=r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\Scratch - ANN\data.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        device="cpu",   # or device=0
        plots=True
    )

if __name__ == "__main__":
    main()