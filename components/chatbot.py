import google.generativeai as genai

def initialize_gemini(api_key):
    """Initialize Gemini AI with the provided API key"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-001')
        return model
    except Exception as e:
        return f"Failed to initialize Gemini AI: {e}", "error"
        

def generate_medical_response(model, conversation_history, user_prompt):
    """Generate a medical response based on conversation history and user prompt"""
    # Create a comprehensive prompt that includes context
    full_prompt = f"""You are a helpful AI medical assistant. 
    Provide professional, empathetic, and informative medical guidance.
    Always prioritize safety and recommend consulting a healthcare professional.

    Conversation History:
    {conversation_history}

    Latest User Query:
    {user_prompt}

    Your response should be:
    - Clear and easy to understand
    - Providing helpful medical information
    - Cautious and avoiding definitive diagnoses
    - Encouraging professional medical consultation
    """

    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"        