import os
import cv2

# =======================
# PATHS
# =======================
ROOT = r"C:\Users\hp\Desktop\VS Code Projects\VS_AllCodes\Scratch - ANN\dataset\val"
IMAGES_DIR = os.path.join(ROOT, "images")
MASTER_LABEL = os.path.join(ROOT, "label.txt")
LABELS_DIR = os.path.join(ROOT, "labels")

os.makedirs(LABELS_DIR, exist_ok=True)

# =======================
# Helper
# =======================
def to_yolo(x, y, w, h, img_w, img_h):
    x_c = (x + w / 2) / img_w
    y_c = (y + h / 2) / img_h
    w_n = w / img_w
    h_n = h / img_h
    return x_c, y_c, w_n, h_n


# =======================
# Parse master label file
# =======================
with open(MASTER_LABEL, "r") as f:
    lines = f.readlines()

current_image = None
img_w = img_h = None
yolo_lines = []

for line in lines:
    line = line.strip()

    # New image block
    if line.startswith("#"):
        # Save previous image labels
        if current_image and yolo_lines:
            out_path = os.path.join(
                LABELS_DIR,
                current_image.replace(".jpg", ".txt")
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            with open(out_path, "w") as out:
                out.writelines(yolo_lines)

        # Reset
        current_image = line[1:].strip()
        yolo_lines = []

        img_path = os.path.join(IMAGES_DIR, current_image)
        if not os.path.exists(img_path):
            current_image = None
            continue

        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

    else:
        if current_image is None:
            continue

        parts = line.split()
        if len(parts) < 4:
            continue

        # ONLY FIRST 4 NUMBERS (as requested)
        x, y, w, h = map(float, parts[:4])

        # Ignore invalid boxes
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            continue

        x_c, y_c, w_n, h_n = to_yolo(x, y, w, h, img_w, img_h)

        # Clamp safety
        x_c = min(max(x_c, 0), 1)
        y_c = min(max(y_c, 0), 1)
        w_n = min(max(w_n, 0), 1)
        h_n = min(max(h_n, 0), 1)

        yolo_lines.append(
            f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n"
        )

# Save last image
if current_image and yolo_lines:
    out_path = os.path.join(
        LABELS_DIR,
        current_image.replace(".jpg", ".txt")
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as out:
        out.writelines(yolo_lines)

print("✅ Labels folder created using ONLY label.txt (first 4 numbers used)")
