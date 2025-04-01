import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import time
import traceback
from components.cavity import cavity_detection
from components.brain import brain_tumor_detection
from components.face import face_stress_analysis
from components.insights import generate_health_insights
from components.model import load_yolo_model
from components.skin import skin_condition_detection
from components.chatbot import initialize_gemini, generate_medical_response

st.set_page_config(page_title="Med-AI --> Medical Diagnosis Assistant", layout="wide")

def process_live_detection(detection_type):
    try:
        if detection_type == "Face Stress Analysis":
            model = load_yolo_model("facedetection.pt")
            detection_label = "Face"
        elif detection_type == "Skin Condition Detection":
            model = load_yolo_model("skin.pt")
            detection_label = "Skin Condition"
        
        else:
            st.error("Invalid detection type")
            return
            
        if model is None:
            st.error(f"Failed to load {detection_type} model")
            return
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to access webcam")
            return
            
        st.frame = st.empty()
        stop_button_placeholder = st.empty()
        stop_button = stop_button_placeholder.button("Stop Live Detection", key="stop_live_detection")
        
        frame_skip = 0  # Process every other frame
        running = True
        
        while cap.isOpened() and running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break
                
            # Skip frames for better performance
            frame_skip += 1
            if frame_skip % 2 != 0:
                continue
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize for better performance
            height, width = frame_rgb.shape[:2]
            scale = 0.5
            resized = cv2.resize(frame_rgb, (int(width*scale), int(height*scale)))
            
            results = model.predict(resized, conf=0.25)
            
            # Draw results on original frame
            for r in results:
                if hasattr(r, 'boxes'):
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, (box.xyxy[0] / scale))  # Scale back to original size
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        conf = float(box.conf[0])
                        
                        # Display class name for skin or jaundice detection
                        if detection_type in ["Skin Condition Detection", "Jaundice Detection"] and hasattr(model, 'names'):
                            cls = int(box.cls[0])
                            class_name = model.names[cls] if cls in model.names else f"Class {cls}"
                            label = f"{class_name}: {conf:.2f}"
                        else:
                            label = f"{detection_label}: {conf:.2f}"
                            
                        cv2.putText(frame, label, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            st.frame.image(frame, channels="BGR", use_container_width=True)
            
            # Check if stop button is pressed
            if stop_button:
                running = False
                break
                
            time.sleep(0.03)  # Small delay to prevent UI freezing
            
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        st.error(f"Live detection error: {str(e)}")
        traceback.print_exc()
def main_app():
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f8ff;
            color: #333;
            font-family: sans-serif;
        }
        .stApp {
            max-width: 100%;
            margin: 0 auto;
            padding: 20px;
        }
        body {
            background-color: #f0f8ff; /* Light blue background */
            color: #333;
            font-family: sans-serif;
        }
        .stApp {
            width: 100%;
            margin: 0 auto;
            padding: 20px;
        }
        .st-tabs > div > div > div {
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton > button {
            background-color: #4CAF50; /* Green button */
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #3e8e41;
        }
        .stFileUploader > div > div {
            background-color: #e6f7ff; /* Lighter blue for file uploader */
            border: 1px dashed #4CAF50;
            padding: 15px;
            border-radius: 5px;
        }
        .stRadio > div > label {
            margin-right: 15px;
        }
        .stTextArea > div > div > textarea {
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
        }
        .stSlider > div > div > div > div {
            background-color: #4CAF50;
        }
        .stSelectbox > div > div > div > div {
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .stMultiSelect > div > div > div > div {
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .stMarkdown > p {
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("🩺 MediAI- Medical Diagnosis Assistant")
    st.write("Upload images for analysis and get personalized health insights.")
    
    tab1, tab2, tab3 = st.tabs(["Image Analysis", "Medical Chatbot", "Health Questionnaire"])
    
    with tab1:
        st.header("Medical Image Analysis")
        col1, col2 = st.columns([3, 2])
        
        with col1:
            uploaded_image = st.file_uploader("Upload a medical image (teeth/face/skin/MRI)", type=["jpg", "png", "jpeg"], key="image_uploader")
            
            if uploaded_image:
                try:
                    image = Image.open(uploaded_image)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    image_np = np.array(image)
                    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    analysis_type = st.radio("Select Analysis Type", 
                                            ["Cavity Detection", "Face Stress Analysis", 
                                            "Skin Condition Detection", "Jaundice Detection",
                                            "Brain Tumor MRI Analysis"], 
                                            key="analysis_type_radio")
                    
                    if st.button("Analyze Image", key="analyze_image_btn"):
                        with st.spinner("Analyzing image..."):
                            if analysis_type == "Cavity Detection":
                                result ,status= cavity_detection(image_cv)
                            elif analysis_type == "Face Stress Analysis":
                                result,status = face_stress_analysis(image_cv)
                            elif analysis_type == "Skin Condition Detection":
                                result,status = skin_condition_detection(image_cv)
                            else:  # Brain Tumor MRI Analysis
                                result, status = brain_tumor_detection(image_cv)
                            if status=="success":
                                st.success(f"Analysis Result: {result}")
                            elif status=="error":
                                st.error(f" {result}")  
                            elif status=="warning":
                                st.warning(f"{result}")
                            else:
                                st.write(f"{result}")
                            
                            # Show confidence visualization for detected conditions
                            if "detected" in result.lower() and not result.startswith("✅"):
                                st.subheader("Detection Visualization")
                                # Create a visualization of the analysis
                                try:
                                    if analysis_type == "Brain Tumor MRI Analysis":
                                        # For MRI, apply contour detection to visualize
                                        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                                        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        
                                        # Draw all contours in red color
                                        viz_img = image_cv.copy()
                                        cv2.drawContours(viz_img, contours, -1, (0, 0, 255), 2)
                                        
                                        # Convert back to RGB for display
                                        viz_img_rgb = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)
                                        st.image(viz_img_rgb, caption="Areas of Interest", use_container_width=True)
                                    else:
                                        # Use the model to get boxes
                                        if analysis_type == "Cavity Detection":
                                            model = load_yolo_model("cavity.pt")
                                        elif analysis_type == "Face Stress Analysis":
                                            model = load_yolo_model("facedetection.pt")
                                        elif analysis_type == "Skin Condition Detection":
                                            model = load_yolo_model("skin.pt")
                                        else:  # Jaundice Detection
                                            model = load_yolo_model("jaundice.pt")
                                            
                                        if model:
                                            results = model.predict(image_cv, conf=0.25)
                                            viz_img = image_cv.copy()
                                            
                                            for r in results:
                                                if hasattr(r, 'boxes'):
                                                    for box in r.boxes:
                                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                                        conf = float(box.conf[0])
                                                        
                                                        # Color based on confidence: red (low) to green (high)
                                                        color = (0, int(255 * conf), int(255 * (1 - conf)))
                                                        
                                                        cv2.rectangle(viz_img, (x1, y1), (x2, y2), color, 2)
                                                        
                                                        # Add label with confidence
                                                        if hasattr(model, 'names') and hasattr(box, 'cls'):
                                                            cls = int(box.cls[0])
                                                            class_name = model.names[cls] if cls in model.names else f"Class {cls}"
                                                            label = f"{class_name}: {conf:.2f}"
                                                        else:
                                                            label = f"Confidence: {conf:.2f}"
                                                            
                                                        cv2.putText(viz_img, label, (x1, y1-10), 
                                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                            
                                            # Convert back to RGB for display
                                            viz_img_rgb = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)
                                            st.image(viz_img_rgb, caption="Detection Visualization", use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not generate visualization: {str(e)}")
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
        
        with col2:
            st.subheader("Live Analysis")
            live_detection_type = st.radio("Select Live Detection Type", 
                                        ["Face Stress Analysis", "Skin Condition Detection", "Jaundice Detection"],
                                        key="live_detection_type_radio")
            
            if st.button("Start Live Detection", key="start_live_btn"):
                process_live_detection(live_detection_type)
            
            # Add information about loading the model
            st.info("ℹ️ For Brain Tumor detection, please upload an MRI scan image and select 'Brain Tumor MRI Analysis' in the analysis type.")
            
            # Add model information
            st.subheader("Model Information")
            st.markdown("""
            - **Cavity Detection**: Trained on dental images to identify potential cavities
            - **Face Stress Analysis**: Analyzes facial features for signs of stress
            - **Skin Condition Detection**: Identifies common skin conditions
            - **Jaundice Detection**: Enhanced with color analysis to detect yellowing of skin/eyes
            - **Brain Tumor MRI**: Analyzes brain MRI scans for abnormalities and potential tumors
            """)
            
            st.caption("Note: Ensure MRI scans are clear and properly oriented for best results")
    
    with tab2:
        st.header("💬 Medical AI Chatbot")
        st.info("Describe your symptoms or ask health-related questions. Remember this is for informational purposes only and not a substitute for professional medical advice.")
        
        api_key=os.getenv("GOOGLE_API_KEY")
        
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = ""
        if 'model' not in st.session_state:
            st.session_state.model =  None
        if api_key:
            st.session_state.model = initialize_gemini(api_key)
        user_prompt = st.chat_input("Ask a medical question...", )
        
        if user_prompt and st.session_state.model:
            st.session_state.conversation_history += f"\nUser: {user_prompt}"

        # Display user message
        with st.chat_message("user"):
            st.write(user_prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = generate_medical_response(
                    st.session_state.model, 
                    st.session_state.conversation_history, 
                    user_prompt
                )
                st.write(response)

        st.session_state.conversation_history += f"\nAI Assistant: {response}"
        st.markdown("---")
        st.warning("""
    ⚠️ Important Medical Disclaimer:
    - This AI assistant provides general health information only
    - It is NOT a substitute for professional medical advice
    - Always consult a licensed healthcare provider for diagnosis and treatment
    """)

    
    with tab3:
        st.header("📝 Health & Lifestyle Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            food_habits = st.selectbox("How often do you consume sugary foods?", 
                                        ["Rarely", "Occasionally", "Frequently", "Daily"],
                                        key="food_habits_select")
            sleep_hours = st.slider("Average sleep per night (hours)", 4, 10, 7, key="sleep_slider")
            stress_level = st.radio("How often do you feel stressed?", 
                                    ["Rarely", "Sometimes", "Often", "Always"],
                                    key="stress_radio")
            exercise = st.selectbox("How often do you exercise?",
                                    ["Never", "1-2 times/week", "3-4 times/week", "5+ times/week"],
                                    key="exercise_select")
            
        with col2:
            water_intake = st.selectbox("Daily water intake", 
                                        ["Less than 1L", "1-2L", "More than 2L"],
                                        key="water_intake_select")
            screen_time = st.slider("Daily screen time (hours)", 1, 16, 8, key="screen_time_slider")
            medical_conditions = st.multiselect("Any existing medical conditions?",
                                                ["None", "Diabetes", "Hypertension", "Asthma", "Allergies", "Other"],
                                                key="medical_conditions_multi")
        
        if st.button("Generate Health Insights", key="gen_insights_btn"):
            with st.spinner("Analyzing your health data..."):
                insights = generate_health_insights(food_habits, sleep_hours, stress_level)
                
                st.subheader("Your Health Insights")
                for insight in insights:
                    st.markdown(insight)
                
                st.info("Remember: These are general insights based on limited information. For personalized advice, consult healthcare professionals.")
    
    # Add disclaimer at the bottom
    st.markdown("---")
    st.caption("**Disclaimer:** This application is for educational and informational purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment.")
    


def main():
    
    if "page" not in st.session_state:
        st.session_state.page = "landing"

    if st.session_state.page == "landing":
        main_app()
if __name__ == "__main__":
    main()
        