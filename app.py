import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Classifier")
st.write("Upload a photo of a plant leaf to analyze its health status.")

# 1. EXACT 38-CLASS LABELS FROM YOUR DATASET
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

@st.cache_resource
def load_pure_model():
    # Automatically locate the absolute directory where app.py lives
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'plant_disease_weights.pth')
    
    # weights_only=False allows PyTorch to load Sequential structure containers safely
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()  # Set the model to evaluation/inference mode
    return model

try:
    model = load_pure_model()

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        # Convert uploaded data into a standard RGB Image
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded leaf preview", width=350)

        if st.button("Predict disease"):
            with st.spinner("Analyzing image patterns..."):
                # 2. IMAGE PREPROCESSING (Matches Google Colab / FastAI defaults)
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                # Transform image and expand dimensions to a batch (1, C, H, W)
                img_tensor = preprocess(img).unsqueeze(0)

                # 3. RUN INFRASTRUCTURE INFERENCE PASS
                with torch.no_grad():
                    outputs = model(img_tensor)
                    # Extract the first image item in the batch array slice [0]
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    confidence, class_idx = torch.max(probabilities, dim=0)
                
                # Extract classification texts and confidence values
                prediction = CLASS_NAMES[class_idx.item()]
                confidence_score = confidence.item()

            # Human-readable label formatting for output visibility
            clean_prediction = prediction.replace("___", " - ").replace("_", " ")
            st.success(f"**Result:** {clean_prediction}")
            st.info(f"**Confidence:** {confidence_score:.2%}")

except Exception as e:
    st.error(f"Could not load or run the model: {e}")
