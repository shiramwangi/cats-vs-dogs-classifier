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
label_map_path = st.sidebar.text_input("Label map path", value="checkpoints_corrected/label_map.json")
weights_path = st.sidebar.text_input("Weights path", value="checkpoints_corrected/model_best.pt")
image_size = st.sidebar.number_input("Image size", min_value=64, max_value=512, value=128, step=16)
max_total_mb = st.sidebar.number_input("Max total upload MB", min_value=1, max_value=1024, value=50, step=10)

# Sidebar: Model info (if logs available)
st.sidebar.markdown("---")
st.sidebar.subheader("Model info")
try:
    log_path = Path(weights_path).parent / "train_log.csv"
    if log_path.exists():
        df_log = pd.read_csv(log_path)
        best_row = df_log.iloc[df_log['accuracy'].idxmax()]
        st.sidebar.write(f"Best accuracy: {best_row['accuracy']*100:.2f}% (epoch {int(best_row['epoch'])})")
        st.sidebar.write(f"Best F1: {best_row['f1']*100:.2f}%")
    else:
        st.sidebar.write("Training log not found.")
except Exception:
    st.sidebar.write("Model metrics unavailable.")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    total_bytes = sum(len(f.getvalue()) for f in uploaded_files)
    if total_bytes > max_total_mb * 1024 * 1024:
        st.error(f"Total upload size exceeds {max_total_mb} MB. Please upload fewer/smaller images.")
    else:
        # Load label map robustly (supports {class:idx} or {"0":"cat"})
        raw_map = load_label_map(Path(label_map_path))
        if all(isinstance(k, str) and k.isdigit() for k in raw_map.keys()):
            idx_to_class = {int(k): v for k, v in raw_map.items()}
        else:
            idx_to_class = {v: k for k, v in raw_map.items()}

        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        model = load_model(arch, num_classes=len(idx_to_class), weights_path=Path(weights_path))
        model.eval()

        results = []

        with st.spinner("Predicting..."):
            for file in uploaded_files:
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    pred_idx = int(torch.argmax(probs).item())
                    pred_class = idx_to_class[pred_idx]
                    confidence = float(probs[pred_idx].item())

                # Emoji/icon for top class
                emoji = "🐱" if pred_class.lower().startswith("cat") else "🐶"

                st.markdown(f"### {emoji} {pred_class} ({confidence:.2%}) — {file.name}")
                st.image(image, use_column_width=True)

                # Per-image probability chart
                prob_df = pd.DataFrame({
                    "class": [idx_to_class[i] for i in range(len(idx_to_class))],
                    "probability": [float(probs[i].item()) for i in range(len(idx_to_class))]
                })
                prob_df = prob_df.set_index("class")
                st.bar_chart(prob_df)
                st.markdown("---")

                results.append({
                    "filename": file.name,
                    "prediction": pred_class,
                    "confidence": confidence,
                    **{f"prob_{idx_to_class[i]}": float(probs[i].item()) for i in range(len(idx_to_class))}
                })

        df = pd.DataFrame(results)
        st.subheader("Results table")
        st.dataframe(df, use_container_width=True)

        # Download CSV
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
