import ultralytics

def load_yolo_model(model_path):
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except Exception as e:
        return f"Error loading YOLO model: {e}", "error"
        