"""Streamlit demo: drop in a photo, get a breed prediction and a Grad-CAM map.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_transforms  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
from src.gradcam import GradCAM, find_target_layer, overlay  # noqa: E402
from src.utils import get_device, pretty_class_name  # noqa: E402

st.set_page_config(page_title="Pet Breed Classifier", page_icon="🐾", layout="centered")


@st.cache_resource(show_spinner="Loading the model…")
def load(checkpoint: str):
    device = get_device()
    model, classes, ckpt = load_checkpoint(checkpoint, device)
    _, eval_tf = build_transforms(ckpt.get("image_size", 224))
    return model, classes, eval_tf, device, ckpt


st.title("🐾 Pet breed classifier")
st.caption(
    "Fine-grained classification over the 37 breeds of the Oxford-IIIT Pet dataset. "
    "ResNet-34 pretrained on ImageNet, fine-tuned in two stages."
)

with st.sidebar:
    st.header("Model")
    checkpoint = st.text_input("Checkpoint path", value="runs/resnet34/best.pt")
    show_cam = st.checkbox("Show Grad-CAM", value=True)
    st.markdown(
        "Grad-CAM highlights the pixels that drove the prediction. If the heat "
        "sits on the background rather than the animal, the model is right for "
        "the wrong reason."
    )

if not Path(checkpoint).exists():
    st.warning(
        f"No checkpoint at `{checkpoint}`. Train one first:\n\n"
        "```bash\npython -m src.train --model resnet34 --epochs 12\n```"
    )
    st.stop()

model, classes, eval_tf, device, ckpt = load(checkpoint)
st.sidebar.success(f"{ckpt['arch']} · validation accuracy {ckpt['val_acc']:.1%}")

uploaded = st.file_uploader("Upload a cat or dog photo", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a photo to get a prediction.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
tensor = eval_tf(image).to(device)

with torch.no_grad():
    probs = F.softmax(model(tensor.unsqueeze(0)), dim=1)[0].cpu()

top = probs.topk(5)
best_name = pretty_class_name(classes[top.indices[0]])
best_prob = top.values[0].item()

left, right = st.columns(2)
with left:
    st.image(image, caption="input", use_container_width=True)
with right:
    if show_cam:
        with GradCAM(model, find_target_layer(model)) as cam_fn:
            cam, _, _ = cam_fn(tensor.clone())
        st.image(overlay(tensor, cam), caption="Grad-CAM", use_container_width=True)

st.subheader(f"{best_name} — {best_prob:.1%}")
if best_prob < 0.5:
    st.warning(
        "Low confidence. Either the photo is unlike the training data, or the "
        "breed is one of the pairs the model genuinely confuses."
    )

st.markdown("**Top 5**")
for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
    st.progress(prob, text=f"{pretty_class_name(classes[idx])} — {prob:.1%}")

st.caption(
    "The model only knows these 37 breeds. Give it a rabbit and it will still "
    "answer with a cat or a dog — softmax always sums to one."
)
