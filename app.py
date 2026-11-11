import gradio as gr
from fastai.vision.all import *
import pathlib
from PIL import Image
import tempfile
import os
import numpy as np

pathlib.WindowsPath = pathlib.PosixPath
learn = load_learner("steak_classifier_clean.pkl", cpu=True)

def classify(img):
    if img is None:
        return "No image"
    try:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img.save(f.name)
            p = f.name
        
        out = learn.predict(p)
        os.remove(p)
        return str(out[0])
    except Exception as e:
        return str(e)

gr.Interface(fn=classify, inputs="image", outputs="text").launch()