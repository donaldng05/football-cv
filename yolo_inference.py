from ultralytics import YOLO

model = YOLO("models/best.pt")  # Load the best pre-trained model

results = model.predict("input_videos/08fd33_4.mp4", save=True)
print(results[0])

print("Detected boxes:")
for box in results[0].boxes:
    print(box)
