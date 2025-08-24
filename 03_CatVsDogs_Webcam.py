from ultralytics import YOLO
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import joblib
import torch.nn as nn
import torchvision.models as models

# ------------------------------
# Load YOLO detector (pretrained on COCO)
# ------------------------------
detector = YOLO("yolov8n.pt")  # detects cats/dogs + more

# ------------------------------
# Load your SVM
# ------------------------------
svm = joblib.load("./svm_resnet.pkl")

# CNN backbone
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
efficientnet.classifier = nn.Identity()
efficientnet = efficientnet.to(device).eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = ["Cat", "Dog"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects with YOLO
    results = detector(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])  # YOLO class index
        conf = float(box.conf[0])
        if cls_id in [15, 16]:  # 15 = cat, 16 = dog in COCO
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop region
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Preprocess for SVM
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_tensor = transform(transforms.ToPILImage()(crop_rgb)).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = efficientnet(img_tensor).cpu().numpy()

            pred = svm.predict(feat)[0]
            label = class_names[pred]

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Cat vs Dog with YOLO + SVM", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
