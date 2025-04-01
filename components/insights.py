def generate_health_insights(food_habits, sleep_hours, stress_level):
    insights = []
    
    # Food habits insights
    if food_habits == "Frequently" or food_habits == "Daily":
        insights.append("⚠️ Your frequent consumption of sugary foods increases risk of dental issues. Consider reducing sugar intake.")
    else:
        insights.append("✅ Your limited sugar consumption is good for dental health.")
        
    # Sleep insights
    if sleep_hours < 6:
        insights.append("⚠️ You may not be getting enough sleep. Aim for 7-9 hours for optimal health.")
    elif sleep_hours > 9:
        insights.append("ℹ️ You're sleeping more than average. While extra rest can be good, consistently oversleeping might be worth discussing with a healthcare provider.")
    else:
        insights.append("✅ Your sleep duration is in the healthy range.")
        
    # Stress insights
    if stress_level in ["Often", "Always"]:
        insights.append("⚠️ Your stress levels are concerning. Consider stress-reduction techniques like meditation, exercise, or speaking with a mental health professional.")
    
    # Combined insights
    if food_habits in ["Frequently", "Daily"] and stress_level in ["Often", "Always"]:
        insights.append("⚠️ The combination of high sugar intake and frequent stress can impact both oral and mental health.")
    
    return insights     