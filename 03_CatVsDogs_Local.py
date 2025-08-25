import os
import cv2
import torch
import joblib
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
from PIL import Image



detector = YOLO("yolov8n.pt")

svm = joblib.load("./svm_resnet.pkl")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
efficientnet.classifier = nn.Identity()
efficientnet = efficientnet.to(device).eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = ["Cat", "Dog"]



def classify_and_show(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not read {image_path}")
        return

    results = detector(img, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        
        if cls_id in [15, 16]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            
            with torch.no_grad():
                feat = efficientnet(img_tensor).cpu().numpy()

            
            pred = svm.predict(feat)[0]
            label = class_names[pred]

            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    
    cv2.imshow("Cat vs Dog Classifier", img)
    print(f"Showing: {image_path}")

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()



image_folder = "./images" 

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        classify_and_show(os.path.join(image_folder, filename))
