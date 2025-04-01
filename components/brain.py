from components.model import load_yolo_model
import cv2
import numpy as np

def brain_tumor_detection(image, model=None):
    try:
        if model is None:
            model = load_yolo_model("braintumor.pt")
            
        if model is None:
            return "⚠️ Brain tumor detection model could not be loaded."
            
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
                highest_conf = max([d['confidence'] for d in detections])
                classes_found = set([d["class"] for d in detections])
                
                # More detailed response based on confidence
                if highest_conf > 0.7:
                    return f"⚠️ High confidence ({highest_conf:.2f}) detection of potential brain abnormalities: {', '.join(classes_found)}. Immediate medical consultation is strongly recommended."
                elif highest_conf > 0.4:
                    return f"⚠️ Moderate confidence ({highest_conf:.2f}) detection of potential brain abnormalities: {', '.join(classes_found)}. Medical consultation is recommended."
                else:
                    return f"⚠️ Low confidence ({highest_conf:.2f}) detection of potential brain abnormalities: {', '.join(classes_found)}. Consider follow-up imaging and consultation."
        
        # Additional image analysis for brain MRI
        try:
            # Convert to grayscale if not already
            if len(image.shape) > 2 and image.shape[2] == 3:
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = image
                
            # Apply histogram equalization to enhance contrast
            equalized = cv2.equalizeHist(gray_img)
            
            # Apply threshold to highlight potential abnormalities
            _, thresh = cv2.threshold(equalized, 200, 255, cv2.THRESH_BINARY)
            
            # Count white pixels which might indicate abnormalities
            white_pixel_percentage = (np.sum(thresh > 0) / thresh.size) * 100
            
            if white_pixel_percentage > 10:  # Threshold can be adjusted
                return f"⚠️ Image analysis shows elevated bright regions ({white_pixel_percentage:.2f}%) which may indicate abnormalities. Consider medical consultation.","warning"
        except Exception as e:
            return f"Image analysis failed, relying solely on model detection. Error: {str(e)}" , "warning"
        
        return "✅ No significant brain abnormalities detected in this MRI scan."
        
    except Exception as e:
        return f"Error in brain tumor detection: {str(e)}","error"
        return "⚠️ Analysis failed. Please try again with a clearer MRI image."