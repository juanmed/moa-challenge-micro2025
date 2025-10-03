from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x-cls.pt")  # load an official model

# Export the model
model.export(format="onnx")