import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define a function to preprocess the user's input
def preprocess_input(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    image = image.astype('float32')
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define a function to make a prediction on the preprocessed input
def make_prediction(preprocessed_input):
    prediction = model.predict(preprocessed_input)
    class_names = ['trafficlight', 'speedlimit', 'crosswalk', 'stop']
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

# Define the Streamlit app
def app():
    # Add a title
    st.title("Traffic Sign Recognition")

    # Add a file uploader to allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # If the user has uploaded a file, display the image and make a prediction
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image and make a prediction
        preprocessed_input = preprocess_input(image)
        prediction = make_prediction(preprocessed_input)

        # Display the prediction to the user
        st.write('Prediction: ', prediction)

# Run the Streamlit app
if __name__ == '__main__':
    app()
