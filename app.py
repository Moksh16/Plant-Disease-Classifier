import streamlit as st
import io
import pickle
import torch

# 1. FORCE SAFELY MAPPING CPU STRUCTURES BEFORE FASTAI INTERFERES
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

# Apply the custom patch globally to standard Python unpickling
pickle.Unpickler = CPU_Unpickler

# 2. NOW SAFELY IMPORT FASTAI
from fastai.vision.all import *

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")
st.title("🌿 Plant Disease Classifier")
st.write("Upload a photo of a plant leaf.")

@st.cache_resource
def load_model():
    # Use standard open + pickle loading instead of the cpu=True parameter 
    # to guarantee it strictly bypasses the GPU Resolver lookup matrix
    with open("plant_disease_model.pkl", "rb") as f:
        return pickle.load(f)

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
