import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array
import os
import glob
import zipfile
import tempfile
import h5py

MODEL_PATH="alexnet_best.keras"
NUM_CLASSES=10

CLASS_NAMES=[
    "Babesia_1173", "Leishmania_2701", "Leukocyte_1000X_461",
    "Leukocyte_400X_915", "Plasmodium_843", "RBCs_8995",
    "Toxoplasma_1000X_2933", "Toxoplasma_400X_3758",
    "Trichomonad_10134", "Trypanosome_2385"
]

IMG_SIZE=(224, 224)
DATASET_PATH="Parasite Dataset"

def build_alexnet(num_classes):
    model=Sequential([
        Conv2D(96, (11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
        BatchNormalization(),
        MaxPooling2D((3,3), strides=(2,2)),
        Conv2D(256, (5,5), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((3,3), strides=(2,2)),
        Conv2D(384, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(384, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3,3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((3,3), strides=(2,2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def collect_h5_weights(h5_group, prefix=""):
    weights={}
    for key in h5_group.keys():
        item=h5_group[key]
        full_key=f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            weights[full_key]=np.array(item)
        else:
            weights.update(collect_h5_weights(item, full_key))
    return weights

def load_weights_from_keras3(model, weight_path):
    with h5py.File(weight_path, 'r') as f:
        all_weights=collect_h5_weights(f)
    weight_values=list(all_weights.values())
    model_weights=model.weights
    if len(weight_values)==len(model_weights):
        for mw, wv in zip(model_weights, weight_values):
            mw.assign(wv)
    else:
        idx=0
        for layer in model.layers:
            layer_weights=layer.weights
            if not layer_weights:
                continue
            new_vals=[]
            for w in layer_weights:
                if idx < len(weight_values):
                    new_vals.append(weight_values[idx])
                    idx+=1
            if new_vals:
                try:
                    layer.set_weights(new_vals)
                except Exception:
                    pass

@st.cache_resource
def load_alexnet():
    model=build_alexnet(NUM_CLASSES)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    dummy=np.zeros((1, 224, 224, 3))
    model.predict(dummy, verbose=0)
    tmpdir=tempfile.mkdtemp()
    with zipfile.ZipFile(MODEL_PATH, 'r') as z:
        z.extractall(tmpdir)
    weight_path=os.path.join(tmpdir, "model.weights.h5")
    load_weights_from_keras3(model, weight_path)
    return model

@st.cache_data
def load_reference_images():
    reference={}
    for cls in CLASS_NAMES:
        folder=os.path.join(DATASET_PATH, cls)
        if os.path.exists(folder):
            imgs=glob.glob(os.path.join(folder, "*.jpg"))+glob.glob(os.path.join(folder, "*.png"))+glob.glob(os.path.join(folder, "*.jpeg"))
            if imgs:
                reference[cls]=imgs[0]
    return reference

def preprocess(image):
    image=image.convert("RGB")
    image=image.resize(IMG_SIZE)
    arr=img_to_array(image)/255.0
    arr=np.expand_dims(arr, axis=0)
    return arr

st.set_page_config(page_title="Microorganism Identification", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #000000;
    }

    .stApp {
        background: #000000;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .header-wrap {
        text-align: center;
        padding: 2.5rem 0 1rem 0;
        position: relative;
    }

    .header-tag {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.3em;
        color: #39ff14;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        opacity: 0.8;
    }

    .header-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 3.2rem;
        line-height: 1.1;
        background: linear-gradient(90deg, #39ff14 0%, #00ff41 60%, #ccff00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }

    .header-sub {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #334155;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }

    .divider-line {
        height: 1px;
        background: linear-gradient(90deg, transparent, #39ff1422, #00aaff44, #39ff1422, transparent);
        margin: 1.5rem 0 2rem 0;
        border: none;
    }

    .panel {
        background: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-radius: 2px;
        padding: 1.6rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }

    .panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #39ff1433, transparent);
    }

    .panel-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.25em;
        color: #39ff14;
        text-transform: uppercase;
        margin-bottom: 1rem;
        opacity: 0.7;
    }

    .result-species {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.6rem;
        background: linear-gradient(135deg, #39ff14, #00aaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        line-height: 1.2;
    }

    .result-confidence-num {
        font-family: 'Space Mono', monospace;
        font-size: 3rem;
        font-weight: 700;
        color: #39ff14;
        line-height: 1;
        margin-bottom: 0.2rem;
    }

    .result-confidence-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.2em;
        color: #334155;
        text-transform: uppercase;
    }

    .bar-track {
        background: #111111;
        border-radius: 0;
        height: 3px;
        width: 100%;
        margin-top: 1rem;
        position: relative;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #39ff14, #00aaff);
        box-shadow: 0 0 8px #39ff1455;
    }

    .rank-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.75rem 0;
        border-bottom: 1px solid #0f0f0f;
    }

    .rank-row:last-child {
        border-bottom: none;
    }

    .rank-num {
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        color: #1a2e0a;
        width: 1.2rem;
        text-align: center;
        flex-shrink: 0;
    }

    .rank-name {
        font-family: 'Syne', sans-serif;
        font-size: 0.85rem;
        color: #94a3b8;
        flex: 1;
        font-weight: 600;
    }

    .rank-name.top {
        color: #e2e8f0;
    }

    .rank-pct {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #39ff14;
        flex-shrink: 0;
    }

    .rank-bar-wrap {
        width: 80px;
        background: #111;
        height: 2px;
        flex-shrink: 0;
    }

    .rank-bar-inner {
        height: 100%;
        background: linear-gradient(90deg, #39ff14, #00aaff);
    }

    .upload-zone {
        border: 1px dashed #1a2a2a;
        border-radius: 2px;
        padding: 2rem;
        text-align: center;
        background: #050505;
        transition: border-color 0.2s;
    }

    .upload-hint {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        color: #1a2e0a;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    .ref-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        letter-spacing: 0.2em;
        color: #1a2e0a;
        text-transform: uppercase;
        text-align: center;
        margin-top: 0.6rem;
    }

    .idle-box {
        border: 1px dashed #0f1f1f;
        border-radius: 2px;
        padding: 4rem 2rem;
        text-align: center;
    }

    .idle-text {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        color: #1a2a2a;
        letter-spacing: 0.2em;
        text-transform: uppercase;
    }

    .stFileUploader label { display: none; }

    section[data-testid="stFileUploadDropzone"] {
        background: #050505 !important;
        border: 1px dashed #1a2a2a !important;
        border-radius: 2px !important;
    }

    section[data-testid="stFileUploadDropzone"] p {
        color: #1e3a3a !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.15em !important;
    }

    section[data-testid="stFileUploadDropzone"] svg {
        fill: #1e3a3a !important;
    }

    .stButton > button {
        background: transparent;
        border: 1px solid #39ff1433;
        color: #39ff14;
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        padding: 0.6rem 1.5rem;
        width: 100%;
        border-radius: 2px;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: #39ff1408;
        border-color: #39ff14;
        box-shadow: 0 0 20px #39ff1411;
    }

    .corner-tl {
        position: absolute;
        top: 8px; left: 8px;
        width: 10px; height: 10px;
        border-top: 1px solid #39ff1433;
        border-left: 1px solid #39ff1433;
    }

    .corner-br {
        position: absolute;
        bottom: 8px; right: 8px;
        width: 10px; height: 10px;
        border-bottom: 1px solid #39ff1433;
        border-right: 1px solid #39ff1433;
    }

    img { border-radius: 2px !important; }

    .stImage { border: 1px solid #111; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-wrap">
    <div class="header-tag">Microscopy Diagnostics System v1.0</div>
    <div class="header-title">Microorganism Identification</div>
    <div class="header-sub">AlexNet Neural Classifier &nbsp;|&nbsp; 10 Parasite Classes</div>
</div>
<hr class="divider-line">
""", unsafe_allow_html=True)

model=load_alexnet()
reference_images=load_reference_images()

col_left, col_right=st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown('<div class="panel"><div class="corner-tl"></div><div class="corner-br"></div><div class="panel-label">Input &mdash; Microscopy Image</div>', unsafe_allow_html=True)
    uploaded_file=st.file_uploader("", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image=Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    else:
        st.markdown('<div class="upload-hint">Drop a slide image to begin analysis</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        run_btn=st.button("Run Classification")
    else:
        run_btn=False

    if uploaded_file and reference_images:
        processed=preprocess(image)
        preds_preview=model.predict(processed, verbose=0)[0]
        top_ref_idx=int(np.argmax(preds_preview))
        top_ref_label=CLASS_NAMES[top_ref_idx]
        if top_ref_label in reference_images:
            st.markdown('<div class="panel" style="margin-top:1rem;"><div class="corner-tl"></div><div class="corner-br"></div><div class="panel-label">Reference Sample</div>', unsafe_allow_html=True)
            ref_img=Image.open(reference_images[top_ref_label])
            st.image(ref_img, use_column_width=True)
            st.markdown(f'<div class="ref-label">{top_ref_label.replace("_", " ")}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    if uploaded_file and run_btn:
        processed=preprocess(image)
        preds=model.predict(processed, verbose=0)[0]
        top_idx=int(np.argmax(preds))
        top_label=CLASS_NAMES[top_idx]
        confidence=float(preds[top_idx])*100
        top3_idx=np.argsort(preds)[::-1][:3]

        st.markdown(f"""
        <div class="panel" style="margin-bottom:1rem;">
            <div class="corner-tl"></div><div class="corner-br"></div>
            <div class="panel-label">Primary Detection</div>
            <div class="result-species">{top_label.replace("_", " ")}</div>
            <div style="margin-top:1.2rem;">
                <div class="result-confidence-num">{confidence:.1f}<span style="font-size:1.2rem; color:#1e3a3a;">%</span></div>
                <div class="result-confidence-label">Confidence Score</div>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:{confidence}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="corner-tl"></div><div class="corner-br"></div><div class="panel-label">Top 3 Candidates</div>', unsafe_allow_html=True)

        for rank, i in enumerate(top3_idx):
            pct=float(preds[i])*100
            name_clean=CLASS_NAMES[i].replace("_", " ")
            bar_w=int(pct)
            is_top="top" if rank==0 else ""
            st.markdown(f"""
            <div class="rank-row">
                <div class="rank-num">0{rank+1}</div>
                <div class="rank-name {is_top}">{name_clean}</div>
                <div class="rank-bar-wrap"><div class="rank-bar-inner" style="width:{bar_w}%;"></div></div>
                <div class="rank-pct">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="idle-box" style="margin-top:0;">
            <div class="idle-text">Awaiting sample input</div>
            <div style="margin-top:0.5rem; font-family:'Space Mono',monospace; font-size:0.55rem; color:#0f1a1a; letter-spacing:0.15em;">Upload an image and run classification</div>
        </div>
        """, unsafe_allow_html=True)



