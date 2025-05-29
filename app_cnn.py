import os
import urllib.request
import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

# Descargar modelo si no existe
modelo_path = "modelo_frutas_cnn.h5"

# Cargar modelo desde la variable correcta
model = load_model(modelo_path)

dclass_names = ["Manzana", "Banana", "Limón", "Naranja", "Pera", "Frutilla", "Tomate"]

# Interfaz Streamlit
st.title("Clasificador de Frutas (CNN)")
st.write("Subí una imagen de una fruta y el modelo la clasificará.")

uploaded_file = st.file_uploader("Elegí una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen
    img = Image.open(uploaded_file)
    st.image(img, caption='Imagen cargada', use_column_width=True)

    # Convertir a RGB para evitar problema de 4 canales
    img = img.convert("RGB")

    # Preprocesamiento
    img = img.resize((100, 100))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Resultado
    st.markdown(f"### 🔍 Resultado: **{predicted_class}**")
