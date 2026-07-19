"""
UC Merced Land Use Classification — Streamlit App
---------------------------------------------------
- Lets the user pick between two trained Keras models (a custom CNN and a
  fine-tuned ResNet).
- Runs inference on an uploaded image.
- Shows Grad-CAM explainability overlay for the predicted class.

Expected files in the same directory as this script:
    best_cnn.h5
    best_resNetFinetuned.h5

Run with:
    streamlit run app.py
"""

import os

# Force TensorFlow to use the legacy (Keras 2) implementation. Models saved
# with older TF/Keras versions (.h5 files where e.g. BatchNormalization's
# `axis` is stored as a list like [3] instead of an int) fail to deserialize
# under standalone Keras 3, which TF >=2.16 uses by default. This must be
# set BEFORE tensorflow is imported. Requires the `tf-keras` package
# (see requirements.txt).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import matplotlib.cm as cm
from PIL import Image

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="UC Merced Land Use Classifier",
    page_icon="🛰️",
    layout="wide",
)

MODEL_PATHS = {
    "Custom CNN (best_cnn.h5)": "best_cnn.h5",
    #"ResNet Fine-tuned (best_resNetFinetuned.h5)": "best_resNetFinetuned.h5",
}

# UC Merced Land Use dataset — 21 classes (alphabetical, standard ordering)
CLASS_NAMES = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
    "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
    "harbor", "intersection", "mediumresidential", "mobilehomepark",
    "overpass", "parkinglot", "river", "runway", "sparseresidential",
    "storagetanks", "tenniscourt",
]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return keras.models.load_model(path, compile=False)


def get_input_size(model) -> tuple[int, int]:
    """Infer expected (H, W) from the model's input shape."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    h, w = shape[1], shape[2]
    # Fall back to a sane default if the model uses a dynamic input size
    h = h if h else 224
    w = w if w else 224
    return int(h), int(w)


def preprocess_image(img: Image.Image, target_size: tuple[int, int]) -> np.ndarray:
    img = img.convert("RGB").resize((target_size[1], target_size[0]))
    arr = np.array(img).astype("float32")
    arr = arr / 255.0
    return arr


def find_last_conv_layer(model) -> str:
    """Walk the model backwards and return the name of the last Conv layer.
    Handles nested models (e.g. a ResNet base wrapped inside a Sequential/
    Functional wrapper) by recursing into sub-models."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.Model):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                continue
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D,
                               keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError("No convolutional layer found in model.")


def locate_layer_path(model, layer_name: str):
    """Return the chain of containers [model, submodel1, ..., owner] such
    that owner.get_layer(layer_name) is a direct child. Returns None if not
    found."""
    for layer in model.layers:
        if layer.name == layer_name:
            return [model]
        if isinstance(layer, keras.Model):
            sub_path = locate_layer_path(layer, layer_name)
            if sub_path is not None:
                return [model] + sub_path
    return None


def build_gradcam_model(model, last_conv_layer_name: str):
    """Build the pieces needed for Grad-CAM without ever re-wiring nested
    sub-models onto a foreign Input (which silently keeps pointing at their
    own original input and causes a 'graph disconnected' error), and
    without manually replaying layers that live INSIDE a branching
    architecture like ResNet (residual 'Add' layers need two inputs, not
    one, so hand-replaying them breaks).

    Strategy:
      1. Find the direct owner (possibly a nested submodel) of the target
         conv layer. That owner is itself a fully self-contained,
         already-built graph.
      2. Ask Keras for BOTH the conv layer's activations AND the owner's
         own final output in one `keras.Model(owner.input, [conv_layer.output,
         owner.output])` call. This lets Keras's real graph -- including any
         skip connections -- handle everything that happens *inside* the
         owner correctly; nothing inside it is hand-replayed.
      3. Only layers OUTSIDE the owner (i.e. the classifier head appended
         after a pretrained base -- pooling / flatten / dense / softmax,
         which is always a plain linear stack) are manually replayed inside
         the GradientTape.
    """
    path = locate_layer_path(model, last_conv_layer_name)
    if path is None:
        raise ValueError(f"Layer {last_conv_layer_name} not found.")

    owner = path[-1]
    conv_layer = owner.get_layer(last_conv_layer_name)

    sub_model = keras.models.Model(owner.input, [conv_layer.output, owner.output])

    # Only layers strictly outside `owner` need manual replay.
    remaining_layers = []
    for i in range(len(path) - 2, -1, -1):
        container = path[i]
        child = path[i + 1]
        idx = container.layers.index(child)
        remaining_layers.extend(container.layers[idx + 1:])

    return sub_model, remaining_layers


def make_gradcam_heatmap(img_array: np.ndarray, model, last_conv_layer_name: str,
                          pred_index: int = None):
    sub_model, remaining_layers = build_gradcam_model(model, last_conv_layer_name)

    with tf.GradientTape() as tape:
        conv_output, base_output = sub_model(img_array[np.newaxis, ...], training=False)
        tape.watch(conv_output)
        x = base_output
        for layer in remaining_layers:
            x = layer(x, training=False)
        predictions = x
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), predictions.numpy()[0]


def overlay_heatmap(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    img = img.convert("RGB")
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_img = Image.fromarray(heatmap_resized).resize(img.size)
    heatmap_arr = np.array(heatmap_img)

    jet = cm.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_arr]
    jet_heatmap = np.uint8(jet_heatmap * 255)
    jet_heatmap_img = Image.fromarray(jet_heatmap).resize(img.size)

    blended = Image.blend(img, jet_heatmap_img, alpha)
    return blended


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("🛰️ UC Merced Land Use Classification")
st.caption("Upload an aerial/satellite image tile, choose a model, and inspect Grad-CAM explainability.")

with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Choose model", list(MODEL_PATHS.keys()))
    alpha = st.slider("Grad-CAM overlay opacity", 0.0, 1.0, 0.45, 0.05)
    top_k = st.slider("Show top-K predictions", 1, 10, 5)
    st.markdown("---")
    st.markdown(
        "- `best_cnn.h5`\n"
       # "- `best_resNetFinetuned.h5`\n\n"
        "UC Merced Land Use dataset — 21 classes."
    )

model_path = MODEL_PATHS[model_choice]
model = load_model(model_path)

if model is None:
    st.error(
        f"Could not find **{model_path}** in the app directory. "
        f"Place your trained `.h5` file next to `app.py` and reload."
    )
    st.stop()

input_size = get_input_size(model)

uploaded_file = st.file_uploader(
    "Upload a land-use image (jpg/png/tif)", type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input image")
        st.image(image, use_container_width=True)

    img_array = preprocess_image(image, input_size)

    with st.spinner("Running inference..."):
        try:
            last_conv = find_last_conv_layer(model)
            heatmap, pred_index, probs = make_gradcam_heatmap(img_array, model, last_conv)
            gradcam_ok = True
        except Exception as e:
            # Still get a prediction even if Grad-CAM fails for some
            # unusual architecture
            probs = model.predict(img_array[np.newaxis, ...], verbose=0)[0]
            pred_index = int(np.argmax(probs))
            gradcam_ok = False
            gradcam_error = str(e)

    pred_class = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) else str(pred_index)
    confidence = float(probs[pred_index]) * 100

    with col2:
        st.subheader("Grad-CAM explanation")
        if gradcam_ok:
            overlay = overlay_heatmap(image, heatmap, alpha=alpha)
            st.image(overlay, use_container_width=True)
        else:
            st.warning(f"Grad-CAM unavailable for this model/layer: {gradcam_error}")

    st.markdown("---")
    st.subheader(f"Prediction: **{pred_class}**  ({confidence:.2f}% confidence)")

    # Top-K bar chart
    order = np.argsort(probs)[::-1][:top_k]
    top_labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in order]
    top_values = [float(probs[i]) for i in order]

    st.bar_chart(
        {"probability": top_values},
        x=None,
        height=300,
        use_container_width=True,
    )
    st.table(
        {"class": top_labels, "probability (%)": [f"{v*100:.2f}" for v in top_values]}
    )
else:
    st.info("Upload an image to run classification and view Grad-CAM.")