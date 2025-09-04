import streamlit as st
from PIL import Image
import io
from pathlib import Path
import pandas as pd
import torch
from torchvision import transforms
from src.infer import load_model
from src.utils import load_label_map

st.set_page_config(page_title="Cats vs Dogs Classifier", page_icon="🐾", layout="centered")

st.title("🐾 Cats vs Dogs Classifier")

st.sidebar.header("Settings")
arch = st.sidebar.selectbox("Model", ["resnet18", "mobilenet_v2", "efficientnet_b0", "baseline"], index=0)
label_map_path = st.sidebar.text_input("Label map path", value="checkpoints/label_map.json")
weights_path = st.sidebar.text_input("Weights path", value="checkpoints/model_best.pt")
image_size = st.sidebar.number_input("Image size", min_value=64, max_value=512, value=128, step=16)
max_total_mb = st.sidebar.number_input("Max total upload MB", min_value=1, max_value=1024, value=50, step=10)

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    total_bytes = sum(len(f.getvalue()) for f in uploaded_files)
    if total_bytes > max_total_mb * 1024 * 1024:
        st.error(f"Total upload size exceeds {max_total_mb} MB. Please upload fewer/smaller images.")
    else:
        label_map = load_label_map(Path(label_map_path))
        idx_to_class = {v: k for k, v in label_map.items()}

        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        model = load_model(arch, num_classes=len(label_map), weights_path=Path(weights_path))
        model.eval()

        results = []
        cols = st.columns(min(3, len(uploaded_files)))

        with st.spinner("Predicting..."):
            for idx, file in enumerate(uploaded_files):
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                with cols[idx % len(cols)]:
                    st.image(image, caption=file.name, use_column_width=True)

                tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    pred_idx = int(torch.argmax(probs).item())
                    pred_class = idx_to_class[pred_idx]
                    confidence = float(probs[pred_idx].item())

                results.append({
                    "filename": file.name,
                    "prediction": pred_class,
                    "confidence": confidence,
                    **{f"prob_{idx_to_class[i]}": float(probs[i].item()) for i in range(len(idx_to_class))}
                })

        df = pd.DataFrame(results)
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)

        # Aggregate confidence distribution chart
        st.subheader("Confidence by class")
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        chart_df = df.melt(id_vars=["filename", "prediction", "confidence"], value_vars=prob_cols, var_name="class", value_name="probability")
        chart_df["class"] = chart_df["class"].str.replace("prob_", "", regex=False)
        st.bar_chart(chart_df.set_index("class")["probability"].groupby(level=0).mean())

        # Download CSV
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
