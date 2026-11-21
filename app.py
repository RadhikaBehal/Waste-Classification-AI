import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Waste Classification AI")
st.markdown("### Intelligent waste detection using deep learning")
st.write("Upload an image to identify its waste category.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_classifier.keras")

def load_classes():
    with open("class_names.json", "r") as f:
        return json.load(f)

try:
    model = load_model()
    classes = load_classes()

    st.success("AI model ready!")

    uploaded_file = st.file_uploader(
        "Upload an image of waste",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Classify Waste", type="primary"):

            resized_image = image.resize((224, 224))

            image_array = np.array(resized_image, dtype=np.float32)
            image_array = np.expand_dims(image_array, axis=0)

            with st.spinner("Analyzing image..."):
                predictions = model.predict(image_array, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = classes[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

            with col2:
                st.subheader("Prediction")
                st.success(f"♻️ Category: {predicted_class.upper()}")
                st.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("Prediction Probabilities")

            for i, category in enumerate(classes):
                score = float(predictions[i])
                st.write(f"**{category.capitalize()}**: {score*100:.2f}%")
                st.progress(score)

            recommendations = {
                "cardboard": "Flatten cardboard boxes and place them in the cardboard recycling bin.",
                "glass": "Separate glass containers for recycling. Handle broken glass carefully.",
                "metal": "Separate metal cans and scrap for recycling.",
                "paper": "Keep paper dry and place it in the paper recycling bin.",
                "plastic": "Clean plastic containers and check local recycling guidelines.",
                "trash": "Dispose of this material in the appropriate non-recyclable waste bin."
            }

            st.subheader("♻️ Disposal Recommendation")

            st.info(
                recommendations.get(
                    predicted_class.lower(),
                    "Dispose of this material according to local waste guidelines."
                )
            )

except Exception as e:
    st.error("An error occurred.")
    st.exception(e)