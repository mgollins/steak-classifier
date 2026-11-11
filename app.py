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
        filename="steak_classifier.pkl",
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
requirements.txt
txt
streamlit
fastai
torch
torchvision
pillow
huggingface_hub
Next steps
Upload this app.py and requirements.txt to your GitHub repository.
Go to share.streamlit.io and deploy the app (select the GitHub repo and app.py).
The app will download the model from your existing Hugging Face Space at startup. Let me know what happens when you deploy.

