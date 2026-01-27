# Face Detection using YOLOv8 (WiderFace)

This project trains a YOLOv8n model on the WiderFace dataset for face detection.

⚠️ Status: Work in Progress (WIP)

---

## 📌 Features
- YOLOv8n model (Ultralytics)
- Custom training on WiderFace dataset
- YOLO-format annotations
- Evaluation pipeline (in progress)

---

## 📂 Dataset
Dataset is not included due to size.

Expected structure:

datasets/widerface/

├── images/

│ ├── train/

│ ├── val/

│ └── test/

└── labels/

├── train/

├── val/

└── test/


You can download WiderFace from below and use "Separate_TrainTest_Dataset" to structure that data into above YOLO requested format:
https://shuoyang1213.me/WIDERFACE/

---

## ⚙️ Setup

```bash
pip install ultralytics
pip install -r requirements.txt
