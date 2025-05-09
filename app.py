import streamlit as st
import pandas as pd
import pickle
import re
from deep_translator import GoogleTranslator
from gtts import gTTS

# Set Streamlit page config
st.set_page_config(page_title="Prakriti Classifier", page_icon="🌿", layout="centered")

# -----------------------
# Load Model and Encoders
# -----------------------
with open("prakriti_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("prakriti_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

features = [f for f in label_encoders if f != "prakriti"]
options = {feature: label_encoders[feature].classes_.tolist() for feature in features}

# -----------------------
# Remedies Dictionary
# -----------------------
remedies = remedies = remedies = {
    'Vata': """
### 🌿 Vata Dosha (Air)
- Common Issues: Dry skin, bloating, anxiety, joint pain, insomnia  
- Balance with: Warm, oily, and grounding foods & habits  

Home Remedies  
- Sesame Oil Massage – Reduces dryness  
- Ginger & Ajwain Tea – Boosts digestion  
- Warm Milk with Nutmeg – Aids better sleep  
- Soaked Almonds – Nourishes nervous system  
- Turmeric & Ghee Mix – Reduces joint pain  
- Avoid: Cold foods, raw vegetables, excessive fasting
""",
    'Pitta': """
### 🔥 Pitta Dosha (Fire)
- Common Issues: Acid reflux, inflammation, irritability, skin rashes  
- Balance with: Cooling, hydrating, and calming remedies  

Home Remedies  
- Aloe Vera Juice – Cools acidity  
- Coconut Water – Naturally hydrating  
- Coriander & Fennel Tea – Soothes digestion  
- Sandalwood Paste – Reduces rashes  
- Cucumber & Mint Smoothie – Cools internal heat  
- Avoid: Spicy foods, fermented foods, caffeine
""",
    'Kapha': """
### 🌍 Kapha Dosha (Earth & Water)
- Common Issues: Weight gain, sluggish digestion, mucus buildup, lethargy  
- Balance with: Light, warm, and stimulating foods  

Home Remedies  
- Honey & Warm Water – Burns excess fat  
- Ginger & Black Pepper Tea – Stimulates metabolism  
- Turmeric & Cinnamon Milk – Boosts immunity  
- Triphala Powder – Detoxifies body  
- Dry Brushing – Improves circulation  
- Avoid: Dairy, fried foods, excessive sweets
"""
}



# -----------------------
# Translation Setup
# -----------------------
languages = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa"
}
selected_lang = st.selectbox("🌐 Choose Language", list(languages.keys()), index=0)
selected_lang_code = languages[selected_lang]

# -----------------------
# Streamlit App UI
# -----------------------
st.title("🌿 Ayurvedic Prakriti Classifier")
st.write("Answer the following questions to determine your **Prakriti (Body Constitution)** and get personalized **home remedies**.")

feature_labels = {
    'gender': 'Gender',
    'bodyDevelopment': 'Body Development',
    'bodyType': 'Body Type',
    'hairColor': 'Hair Color',
    'hairThickness': 'Hair Thickness',
    'eyeColor': 'Eye Color',
    'hungerLevel': 'Hunger Level',
    'constipationTendency': 'Constipation Tendency',
    'weightVariation': 'Weight Variation',
    'sleep': 'Sleep Pattern',
    'physicalStrength': 'Physical Strength',
    'hairGraying': 'Hair Graying',
    'wrinkles': 'Wrinkles',
    'mindStability': 'Mind Stability'
}

with st.form("prakriti_form"):
    user_input = {}
    for feature in features:
        user_input[feature] = st.selectbox(
            label=feature_labels.get(feature, feature),
            options=["Select"] + options[feature],
            key=feature
        )
    submitted = st.form_submit_button("Predict Prakriti")

# -----------------------
# Markdown Stripper for Audio
# -----------------------
def strip_markdown(text):
    text = re.sub(r'\*+', '', text)           # Remove * and **
    text = re.sub(r'#+\s*', '', text)         # Remove ### etc.
    return text.strip()

# -----------------------
# Prediction and Output
# -----------------------
if submitted:
    if "Select" in user_input.values():
        st.warning("⚠️ Please answer all questions before submitting.")
    else:
        input_df = pd.DataFrame([user_input])
        for col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]
        prakriti_type = label_encoders['prakriti'].inverse_transform([prediction])[0]

        st.markdown(f"## 🌿 Predicted Prakriti Type: **{prakriti_type}**")
        original_text = remedies.get(prakriti_type, "❌ No remedies found.")

        try:
            if selected_lang != "English":
                translated_text = GoogleTranslator(source='auto', target=selected_lang_code).translate(text=original_text)
                st.markdown(translated_text)
                clean_for_audio = strip_markdown(translated_text)
                audio = gTTS(text=clean_for_audio, lang=selected_lang_code)
            else:
                st.markdown(original_text)
                clean_for_audio = strip_markdown(original_text)
                audio = gTTS(text=clean_for_audio, lang='en')

            audio.save("remedy.mp3")
            with open("remedy.mp3", "rb") as f:
                st.audio(f.read(), format="audio/mpeg")
        except Exception as e:
            st.warning("⚠️ Translation failed. Showing original content in English.")
            st.markdown(original_text)
            st.text(f"Error: {str(e)}")
