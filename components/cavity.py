from components.model import load_yolo_model

def cavity_detection(image, model=None):
    try:
        if model is None:
            model = load_yolo_model("cavity.pt")
        
        if model is None:
            return "⚠️ Model could not be loaded."
            
        results = model.predict(image, conf=0.25)
        
        # Get detection counts and confidence scores
        detections = []
        if results and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf[0])
                detections.append({
                    "confidence": conf,
                    "coordinates": box.xyxy[0].tolist()
                })
            
            if len(detections) > 0:
                return f"🦷 {len(detections)} potential cavities detected with confidence scores ranging from {min([d['confidence'] for d in detections]):.2f} to {max([d['confidence'] for d in detections]):.2f}. Consider visiting a dentist."
        
        return "✅ No cavities detected. Maintain regular oral hygiene."
        
    except Exception as e:
        return f"Error in cavity detection: {str(e)}" , "error"
        return "⚠️ Analysis failed. Please try again with a clearer image."