
import streamlit as st
from fastai.vision.all import *

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")
st.title("🌿 Plant Disease Classifier")
st.write("Upload a photo of a plant leaf.")

@st.cache_resource
def load_model():
    return load_learner("plant_disease_model.pkl", cpu=True)

try:
    learn = load_model()

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = PILImage.create(uploaded_file)
        st.image(img, caption="Uploaded leaf", width=350)

        if st.button("Predict disease"):
            with st.spinner("Analyzing image..."):
                prediction, idx, probs = learn.predict(img)

            st.success(f"Prediction: {prediction}")
            st.write(f"Confidence: {probs[idx].item():.2%}")

except Exception as e:
    st.error(f"Could not load or run the model: {e}")
