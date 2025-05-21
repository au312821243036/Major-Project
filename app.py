from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# setting page layout
st.set_page_config(
    page_title="Interactive Interface for YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Blood Group Detection")
st.info(f"""### Blood Group Determination:
        A Positive (A+): Agglutination with anti-A and anti-D.
    A Negative (A-): Agglutination with anti-A, no agglutination with anti-D.
    B Positive (B+): Agglutination with anti-B and anti-D.
    B Negative (B-): Agglutination with anti-B, no agglutination with anti-D.
    AB Positive (AB+): Agglutination with anti-A, anti-B, and anti-D.
    AB Negative (AB-): Agglutination with anti-A and anti-B, no agglutination with anti-D.
    O Positive (O+): No agglutination with anti-A or anti-B, agglutination with anti-D.
    O Negative (O-): No agglutination with anti-A, anti-B, or anti-D""")
# sidebar
st.sidebar.header("Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.radio(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = "bloodgroupdetectionmodel.pt"
if model_type:
    model_path = "bloodgroupdetectionmodel.pt"
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
st.sidebar.header("Image Config")
source_selectbox = "Image"
st.sidebar.radio('Source',options=["Image"])
source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")