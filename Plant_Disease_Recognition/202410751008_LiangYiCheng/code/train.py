from ultralytics import YOLO

model = YOLO("yolo11m.pt")
results = model.train(data="./data_example/train/data.yaml", epochs=400)
