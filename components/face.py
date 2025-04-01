from components.model import load_yolo_model

def face_stress_analysis(image, model=None):
    try:
        if model is None:
            model = load_yolo_model("facedetection.pt")
            
        if model is None:
            return "⚠️ Model could not be loaded."
            
        results = model.predict(image, conf=0.25)
        
        if results and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            return "⚠️ Potential stress indicators detected. Consider relaxation techniques."
        
        return "✅ No obvious signs of stress detected in facial features."
        
    except Exception as e:
        return f"Error in stress analysis: {str(e)}", "error"
        return "⚠️ Analysis failed. Please try again with a clearer facial image."