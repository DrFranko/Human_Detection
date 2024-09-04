import cv2
import torch
import torch.nn as nn
import numpy as np
from yolov5 import YOLOv5

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

yolo = YOLOv5("yolov5s.pt")
yolo.classes = [0] #Only Humans

max_cosine_distance = 0.3
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

class AgeClassifier(nn.Module):
    def __init__(self):
        super(AgeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


from dataset import FGNETDataset
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = FGNETDataset(
    root_dir='path/to/fgnet_root_dir/images',
    csv_file='path/to/fgnet_root_dir/fgnet_data.csv',
    transform=transform
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = AgeClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), "age_classification_model.pth")

def load_age_model():
    model = AgeClassifier()
    model.load_state_dict(torch.load("age_classification_model.pth"))
    model.eval()  
    return model

age_model = load_age_model()

def classify_age(image, model):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = transform(image).unsqueeze(0)  

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return "Child" if predicted.item() == 0 else "Adult"

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = yolo(frame)
        boxes = results.xyxy[0].cpu().numpy()
        
        features = gdet.create_box_encoder("mars-small128.pb", batch_size=1)(frame, boxes)
        detections = [Detection(box, 1.0, feature) for box, feature in zip(boxes, features)]
        
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            bbox = track.to_tlbr()
            
            person_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            age_class = classify_age(person_img, age_model)
            
            color = (0, 255, 0) if age_class == "Child" else (0, 0, 255)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, f"{age_class} ID-{track.track_id}", (int(bbox[0]), int(bbox[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()