import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.keras")

st.title("CNN Classifier")
st.markdown("**Optimized | 94% Val Acc**")

uploaded = st.file_uploader("Upload",type=['png','jpg'])
if uploaded:
    img = Image.open(uploaded)
    st.image(img,use_column_width=True)
    
    img = np.array(img)
    img = cv2.resize(img,(64,64))/255.0
    pred = load_model().predict(img[np.newaxis])
    
    st.success(f"**Class {int(np.argmax(pred[0]))}**")
    st.success(f"**{np.max(pred[0])*100:.1f}%**")
    st.progress(np.max(pred[0]))
