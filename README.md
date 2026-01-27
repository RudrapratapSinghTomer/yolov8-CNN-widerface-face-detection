# Face Detection using YOLOv8 (WiderFace)

This project implements a face detection system using **YOLOv8n** trained on the **WiderFace dataset**.

🚧 **Status: Work in Progress**

The current pipeline focuses on training and evaluating a YOLOv8-based face detector.  
In future updates, a **CNN-based face refinement / classification module** will be added to further improve detection accuracy and robustness.

---

## 📌 Features
- YOLOv8n model (Ultralytics)
- Training on WiderFace dataset (YOLO format)
- Modular training and evaluation scripts
- Designed for future CNN integration

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
