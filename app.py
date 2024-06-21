import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st
import matplotlib.pyplot as plt  # Only imported if necessary for visualization


from tensorflow import keras

# from my_custom_module import MyCustomLayer  # Import the necessary module or define the missing class

model = keras.models.load_model('mymodel.keras')  # Replace with your pre-trained image classification model
labels=['glioma','meningioma','notumor','pituitary']  # Replace with your actual class labels

def predict(uploaded_file):
    if model is None:
        st.error("Please load your pre-trained model before making predictions.")
        return
    if uploaded_file is not None:
        img=uploaded_file
        resized_img = img.resize((299, 299))
        img = np.asarray(resized_img)
        img = np.expand_dims(img, axis=0)
        img = img / 255
        predictions = model.predict(img)
        probs = list(predictions[0])
        plt.figure(figsize=(6, 4))
        bars = plt.barh(labels, probs)
        plt.xlabel("Probability", fontsize=12)
        ax = plt.gca()
        ax.bar_label(bars, fmt="%.2f")
        plt.tight_layout()
        st.pyplot()  # Display the matplotlib plot
        
    else:
        raise FileNotFoundError("No file uploaded")


def main():
    """Main function to build and run the Streamlit app."""

    st.title("Image Prediction App")

    uploaded_file = st.file_uploader("Choose an image to upload:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Read the image as a NumPy array
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)
        except Exception as e:
            st.error(f"Error uploading image: {e}")

    if image is not None:
        predict(image)

if __name__ == "__main__":
    main()
