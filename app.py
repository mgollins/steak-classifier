import streamlit as st
from fastai.vision.all import *
import pathlib
from PIL import Image
import tempfile
import os
from huggingface_hub import hf_hub_download

pathlib.WindowsPath = pathlib.PosixPath

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="LabBusters/steak-classifier",
        filename="steak_classifier_colab.pkl",
        repo_type="space"
    )
    return load_learner(model_path, cpu=True)

learn = load_model()

st.title("Steak Check")
st.write("Upload a photo of the steak through the package.")

uploaded = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded photo", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        path = tmp.name

    try:
        pred, idx, probs = learn.predict(path)
        conf = float(probs[idx]) * 100
        st.success(f"**{pred}** ({conf:.1f}% confidence)")
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        os.unlink(path)
