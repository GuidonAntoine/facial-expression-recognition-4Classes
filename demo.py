import torch
import cv2
import numpy as np
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# === PARAMÈTRES ===
CLASSES = ['angry', 'fear', 'happy', 'sad']
MODEL_PATH = 'TEST_final/fine_tuned_model_data_final.pth'  # adapter au besoin

# === CHARGEMENT DU MODÈLE ===
model = models.densenet121(pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print("[OK] Modèle chargé avec succès.")
except Exception as e:
    print("[ERREUR] Impossible de charger le modèle :", e)
    exit()

# === CLASSIFIEUR HAAR POUR VISAGES ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === TRANSFORMATIONS POUR LE MODÈLE ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === CAPTURE WEBCAM ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

print("Webcam détectée. Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture de la frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb).convert('RGB')
        input_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, 1).item()
            label = CLASSES[predicted_class]

        # Dessiner le rectangle vert + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Reconnaissance d'expression faciale", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
