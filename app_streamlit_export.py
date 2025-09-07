import streamlit as st
from pathlib import Path
import io
from PIL import Image
import torch
from torchvision import transforms
import json

try:
    import onnxruntime as ort  # optional for ONNX path
    HAS_ORT = True
except Exception:
    HAS_ORT = False

st.set_page_config(page_title="Cats vs Dogs (Exported Model)", page_icon="🐾", layout="centered")

st.title("🐾 Cats vs Dogs - Exported Model Demo")

st.sidebar.header("Settings")
model_type = st.sidebar.selectbox("Model format", ["TorchScript", "ONNX"], index=0)
image_size = st.sidebar.number_input("Image size", min_value=64, max_value=512, value=128, step=16)

# Default paths
ts_path = st.sidebar.text_input("TorchScript path", value="exports/resnet18.ts.pt")
onnx_path = st.sidebar.text_input("ONNX path", value="exports/resnet18.onnx")
label_map_path = st.sidebar.text_input("Label map path", value="checkpoints_corrected/label_map.json")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_label_map(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    return {v: k for k, v in m.items()}  # idx->class

@st.cache_resource
def load_torchscript(path: Path):
    return torch.jit.load(str(path), map_location="cpu").eval()

@st.cache_resource
def load_onnx_session(path: Path):
    if not HAS_ORT:
        raise RuntimeError("onnxruntime is not installed. pip install onnxruntime")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) 

if uploaded_files:
    try:
        idx_to_class = load_label_map(Path(label_map_path))
    except Exception as e:
        st.error(f"Failed to load label map: {e}")
        st.stop()

    if model_type == "TorchScript":
        try:
            model = load_torchscript(Path(ts_path))
        except Exception as e:
            st.error(f"Failed to load TorchScript model: {e}")
            st.stop()
    else:
        try:
            session = load_onnx_session(Path(onnx_path))
        except Exception as e:
            st.error(f"Failed to load ONNX model: {e}")
            st.stop()

    cols = st.columns(min(3, len(uploaded_files)))
    results = []

    with st.spinner("Predicting..."):
        for i, file in enumerate(uploaded_files):
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            with cols[i % len(cols)]:
                st.image(image, caption=file.name, use_column_width=True)

            x = preprocess(image).unsqueeze(0)

            if model_type == "TorchScript":
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0].tolist()
            else:
                inp = x.numpy()
                outputs = session.run(["logits"], {"input": inp})
                import numpy as np
                logits = outputs[0]
                exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = (exp / exp.sum(axis=1, keepdims=True))[0].tolist()

            pred_idx = int(max(range(len(probs)), key=lambda k: probs[k]))
            pred_class = idx_to_class[pred_idx]
            confidence = probs[pred_idx]

            results.append({
                "filename": file.name,
                "prediction": pred_class,
                "confidence": confidence,
            })

    st.subheader("Results")
    for r in results:
        st.write(f"{r['filename']}: {r['prediction']} ({r['confidence']:.2%})")
