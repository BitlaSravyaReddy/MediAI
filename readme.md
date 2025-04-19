# MediAI - AI-Powered Medical Diagnosis Platform

MediAI is an advanced AI-based medical diagnosis platform designed to help users assess various health concerns using multiple models. The platform uses live detection and image upload features, providing real-time analysis and insights for facial stress, skin type, jaundice detection, cavities, brain tumors, and more. It also integrates with Google Gemini for personalized medical chatbot assistance and generates health insights based on user input.

## Features

1. **Real-Time Detection:**
   - **Facial Stress Detection:** Detects stress levels from facial expressions.
   - **Skin Type Detection:** Identifies skin type (e.g., oily, dry, combination).
   - **Jaundice Detection:** Uses live detection to analyze the presence of jaundice symptoms based on facial features.

2. **Image Upload Analysis:**
   - **Cavity Detection:** Identifies potential cavities in dental images.
   - **Brain Tumor Detection:** Detects potential brain abnormalities from MRI images.

3. **Medical Chatbot Integration:**
   - A chatbot powered by Google Gemini offers personalized health advice, diagnoses, and recommendations.
   - Can answer queries and provide recommendations based on medical prompts.

4. **Health Insights:**
   - Generates health insights based on food habits, sleep hours, and stress levels, providing actionable advice for a healthier lifestyle.


## How to Run

Clone the repository:

git clone https://github.com/BitlaSravyaReddy/MediAI.git
cd mediAI

Install dependencies: Make sure you have the required libraries by running:

pip install -r requirements.txt

Set up Google Gemini API:

Sign up for Google Gemini and get your API key.

Store your API key in a .env file or set it as an environment variable.

## Run the app: 

streamlit run app.py

Open your browser and go to https://medi--ai.streamlit.app/ to start using MediAI.

## Models Used

1. YOLOv10 for Live Detection:

Used for detecting facial stress, skin type, and jaundice in real-time.

YOLOv10 model has been trained to identify key facial features and analyze them accordingly.


2. Google Gemini AI Integration:

The chatbot is integrated with Google Gemini to offer conversational assistance and personalized health advice.

## How It Works

1. Live Detection Models:

Use your webcam to check for stress, skin type, or jaundice.

The system uses YOLOv10 to analyze the live video stream and provides results instantly.

2. Image Upload Analysis:

Upload MRI or dental images for cavity or brain tumor detection.

The system processes the images and returns the analysis with recommendations.

3. Chatbot Interaction:

The AI chatbot, powered by Google Gemini, asks about your concerns and provides helpful responses, guiding users toward possible diagnoses or health-related information.

4. Health Insights Generation:

Input data such as food habits, sleep hours, and stress levels to receive personalized health advice.

## Contributing
If you'd like to contribute to the project, feel free to fork the repository and submit a pull request. Here are some ideas for contributions:

Improving the model's accuracy for cavity and brain tumor detection.

Enhancing the user interface of the platform.

Adding more health analysis features.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

YOLOv10 for real-time object detection.

Google Gemini for chatbot AI integration.

OpenCV for image processing and analysis.

For any inquiries or issues, feel free to open an issue or contact the project maintainers.

Note: This project is in its initial stages and ongoing improvements are being made. Stay tuned for updates!


### Key Highlights:

- **Comprehensive Feature Set:** Covers various aspects of health analysis, from real-time detections to diagnostic assistance via the chatbot.

- **Integration with Google Gemini:** Adds a unique, intelligent, and conversational element to your platform.

- **Image Analysis and Live Detection:** The use of models like YOLOv10 and deep learning for medical image analysis.

