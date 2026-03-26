import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Load model files
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Medical Assistant", page_icon="🩺")

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
<style>
.stButton>button {
    background-color: #2E86C1;
    color: white;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title
# -------------------------------
st.title("🩺 AI Medical Assistant")
st.markdown("### Describe your symptoms in natural language 👇")

st.warning("⚠️ This is not a medical diagnosis. Consult a doctor.")

# -------------------------------
# Confidence Label Function
# -------------------------------
def get_confidence_label(prob):
    if prob > 70:
        return "🟢 High"
    elif prob > 40:
        return "🟡 Medium"
    else:
        return "🔴 Low"

# -------------------------------
# NLP Symptom Extractor
# -------------------------------
def extract_symptoms(user_input):
    user_input = user_input.lower()
    detected = []

    for symptom in columns:
        clean_symptom = symptom.replace("_", " ")
        if clean_symptom in user_input:
            detected.append(symptom)

    return list(set(detected))

# -------------------------------
# Input
# -------------------------------
user_input = st.text_area("💬 Enter symptoms (e.g. fever, headache, vomiting)")

# -------------------------------
# Prediction
# -------------------------------
if st.button("Analyze"):

    if user_input.strip() == "":
        st.error("Please enter symptoms")
    else:
        symptoms = extract_symptoms(user_input)

        if len(symptoms) == 0:
            st.warning("⚠️ No symptoms detected. Try different words.")
            st.stop()

        st.success(f"Detected Symptoms: {', '.join(symptoms)}")

        # -------------------------------
        # MIN SYMPTOMS CHECK
        # -------------------------------
        if len(symptoms) < 3:
            st.warning("⚠️ Please enter at least 3 symptoms for better accuracy")

        # -------------------------------
        # Create input vector
        # -------------------------------
        input_data = [0] * len(columns)

        for symptom in symptoms:
            if symptom in columns:
                index = columns.index(symptom)
                input_data[index] = 1

        input_df = pd.DataFrame([input_data], columns=columns)

        # -------------------------------
        # Prediction
        # -------------------------------
        probs = model.predict_proba(input_df)[0]
        top_prob = max(probs)

        # -------------------------------
        # CONFIDENCE CHECK
        # -------------------------------
        if top_prob < 0.3:
            st.warning("⚠️ Low confidence prediction. Try adding more symptoms.")

        # -------------------------------
        # Top 3 Predictions
        # -------------------------------
        top_indices = probs.argsort()[-3:][::-1]

        diseases = []
        probabilities = []

        st.subheader("🧾 Possible Conditions:")

        for i in top_indices:
            disease = label_encoder.inverse_transform([i])[0]
            probability = probs[i] * 100

            diseases.append(disease)
            probabilities.append(probability)

            label = get_confidence_label(probability)

            st.write(f"👉 {disease} : {probability:.2f}% ({label})")

        st.success("Prediction complete ✅")

        # -------------------------------
        # 📊 DASHBOARD
        # -------------------------------
        st.subheader("📊 Prediction Confidence Dashboard")

        if len(diseases) > 0:
            chart_data = pd.DataFrame({
                "Disease": diseases,
                "Confidence": probabilities
            })

            st.bar_chart(chart_data.set_index("Disease"))

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning & Streamlit")