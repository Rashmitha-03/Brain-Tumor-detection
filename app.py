import streamlit as st
import tensorflow.keras as keras
from PIL import Image, ImageOps
import numpy as np
from img_classification import *

input_shape = (224, 224)
image_types = ["png", "jpg"]

st.set_page_config(
    page_title='Brain Tumor Classification Deep Learning Model',
    layout="wide",
)

def main():
    st.title("Brain Tumor Classification 🧠 ")
    st.header("")
    
    st.write('Select the model to predict input image:')
    option_1 = st.checkbox('VGG16', key='vgg')
    option_2 = st.checkbox('ResNet50', key='resnet')
    
    results_sidebar = st.sidebar.checkbox('Results from trained model')
    if results_sidebar:
        st.sidebar.write("Results VGG16")
        st.sidebar.image("results_vgg.jpg") 
        st.sidebar.write("Results ResNet50")
        st.sidebar.image('results_resnet.jpg')

    if not (option_1 or option_2):
        st.warning("Please select at least one model to upload file for prediction.")
        return

    uploaded_file = st.file_uploader("Choose a brain MRI image for classification as tumor or non-tumor...", type=image_types)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        if option_1:
            st.write("You have chosen VGG16 model for prediction.")
            with st.spinner(text='Classifying using VGG16...'):
                label = binary_classifier_vgg(image, 'model_vgg.h5')
                st.success('Prediction complete.')
            if label == 1:
                st.error("Result: This scan likely shows a tumor. Please consult a doctor.")
            else:
                st.success("Result: No tumor detected. Take good care.")

        if option_2:
            st.write("You have chosen ResNet50 model for prediction.")
            with st.spinner(text='Classifying using ResNet50...'):
                label = binary_classifier_resnet(image, 'model_resnet.h5')
                st.success('Prediction complete.')
            if label == 1:
                st.error("Result: This scan likely shows a tumor. Please consult a doctor.")
            else:
                st.success("Result: No tumor detected. Take good care.")

        if option_1 and option_2:
            st.info("Note: Predictions may vary between models. Please rely on medical advice. No data is stored.")

main()
