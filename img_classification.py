import tensorflow.keras as keras
from PIL import Image, ImageOps
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet

input_shape = (224, 224)

def binary_classifier_vgg(img, model_file):
    model_vgg_loaded = keras.models.load_model(model_file, compile=False)
    image = img.convert('RGB').resize(input_shape, Image.LANCZOS)
    image_array = np.array(image)
    expanded_image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input_vgg(expanded_image_array)
    pred = model_vgg_loaded.predict(preprocessed_image)
    output_label = [1 if x > 0.5 else 0 for x in pred]
    return output_label[0]

def binary_classifier_resnet(img, model_file):
    model_resnet_loaded = keras.models.load_model(model_file, compile=False)
    image = img.convert('RGB').resize(input_shape, Image.LANCZOS)
    image_array = np.array(image)
    expanded_image_array = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input_resnet(expanded_image_array)
    pred = model_resnet_loaded.predict(preprocessed_image)
    output_label = [1 if x > 0.5 else 0 for x in pred]
    return output_label[0]
