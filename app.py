import cv2
import yolov5
import streamlit as st
import numpy as np
import pandas as pd

# Cargar el modelo preentrenado
model = yolov5.load('yolov5s.pt')

# Configurar parámetros del modelo
model.conf = 0.25  # Umbral de confianza NMS
model.iou = 0.45  # Umbral de IoU NMS
model.agnostic = False  # NMS de clase agnóstica
model.multi_label = False  # NMS múltiples etiquetas por cuadro
model.max_det = 1000  # Número máximo de detecciones por imagen

# Título de la aplicación
st.title("Detección de Objetos en Imágenes")

# Barra lateral para configuración de parámetros
with st.sidebar:
    st.header("Configuración del Modelo")
    model.iou = st.slider('Umbral de IoU', 0.0, 1.0, model.iou, 0.01)
    model.conf = st.slider('Nivel de Confianza', 0.0, 1.0, model.conf, 0.01)
    st.info("Ajusta los parámetros para mejorar la detección de objetos.")

# Capturar una imagen con la cámara
st.subheader("Captura de Imagen")
picture = st.camera_input("Haz clic para capturar una foto", label_visibility='visible')

# Procesar la imagen si se ha capturado una
if picture:
    st.success("¡Imagen capturada exitosamente!")

    # Convertir la imagen de bytes a formato OpenCV
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Realizar la inferencia
    with st.spinner("Detectando objetos..."):
        results = model(cv2_img)

    # Mostrar los resultados
    st.subheader("Resultados de la Detección")
    col1, col2 = st.columns([2, 1])

    with col1:
        # Mostrar la imagen con las cajas de detección
        results.render()
        st.image(cv2_img, channels='BGR', caption="Imagen con Detecciones")

    with col2:
        st.subheader("Resumen de Detecciones")

        # Obtener etiquetas y contar categorías detectadas
        label_names = model.names
        category_count = {}
        categories = results.pred[0][:, 5]

        for category in categories:
            category_name = label_names[int(category)]
            category_count[category_name] = category_count.get(category_name, 0) + 1

        # Crear un DataFrame con los resultados
        detections_df = pd.DataFrame(
            [{"Categoría": cat, "Cantidad": count} for cat, count in category_count.items()]
        )

        # Mostrar la tabla con los resultados
        st.table(detections_df.style.format(precision=0).background_gradient(cmap="YlGn"))

        st.markdown(
            f"### Total de objetos detectados: {detections_df['Cantidad'].sum()}",
            unsafe_allow_html=True
        )

    st.success("Detección completada.")
