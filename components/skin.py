from components.model import load_yolo_model
def skin_condition_detection(image, model=None):
    try:
        if model is None:
            model = load_yolo_model("skin.pt")
            
        if model is None:
            return "⚠️ Skin detection model could not be loaded."
            
        results = model.predict(image, conf=0.25)
        
        # Get detection counts and confidence scores
        detections = []
        if results and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls] if hasattr(model, 'names') and cls in model.names else f"Class {cls}"
                
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "coordinates": box.xyxy[0].tolist()
                })
            
            if len(detections) > 0:
                classes_found = set([d["class"] for d in detections])
                return f"🔍 {len(detections)} potential skin conditions detected: {', '.join(classes_found)}. Confidence scores range from {min([d['confidence'] for d in detections]):.2f} to {max([d['confidence'] for d in detections]):.2f}. Consider consulting a dermatologist.","success"
        
        return "✅ No concerning skin conditions detected. Continue regular skin care.", ""
        
    except Exception as e:
        return f"Error in skin condition detection: {str(e)}", "error"
        