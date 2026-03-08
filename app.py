"""
Medical AI Application: Chest X-Ray Pathology Detection
========================================================
A Streamlit-based clinical decision support tool for detecting pathologies
in chest X-ray images using a deep learning model (EfficientNetB4).
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from PIL import Image
import numpy as np
import cv2
import os

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

IMAGE_SIZE = 380
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

AVAILABLE_MODELS = {
    "EfficientNetB4 (Fair - Bias Mitigated)": {
        "path": "./fair-efficientnetb4-pathology-cxr2.pth",
        "description": "Trained with oversampling to reduce bias across clinical subgroups (PA/AP, portable/non-portable)",
        "metrics": {"AUC": 0.749, "Accuracy": 0.712, "F1": 0.645}
    },
    "EfficientNetB4 (Original)": {
        "path": "./efficientnetb4-pathology-cxr.pth",
        "description": "Original model trained on MIMIC-CXR without fairness adjustments",
        "metrics": {"AUC": 0.763, "Accuracy": 0.690, "F1": 0.660}
    },
}

# =============================================================================
# Model Loading
# =============================================================================
@st.cache_resource
def load_model(model_name: str):
    """Load a pre-trained EfficientNetB4 model for pathology classification."""
    model_config = AVAILABLE_MODELS[model_name]
    model_path = model_config["path"]

    model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
    model.classifier[1] = nn.Sequential(nn.Linear(1792, 2))

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        st.warning(f"Model file not found: {model_path}. Using default weights.")

    model.to(DEVICE)
    model.eval()
    return model


def get_last_conv_layer(model):
    """Get the last convolutional layer for GradCAM."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


# =============================================================================
# Image Preprocessing
# =============================================================================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for model inference.

    Transforms:
    - Convert to grayscale, then to 3-channel
    - Resize to 512x512, center crop to 380x380
    - Normalize with ImageNet statistics
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=3),
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(MEAN, STD),
    ])

    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension


def tensor_to_display_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor back to displayable image."""
    img = tensor.squeeze(0).cpu().numpy()
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img * STD + MEAN  # Denormalize
    img = np.clip(img, 0, 1)
    return img


# =============================================================================
# Inference
# =============================================================================
def run_inference(model, image_tensor: torch.Tensor) -> dict:
    """
    Run model inference on preprocessed image.

    Returns:
        dict with prediction, confidence, and probabilities
    """
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        pred_class = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, pred_class].item()

    return {
        "prediction": "Pathology Detected" if pred_class == 1 else "No Pathology",
        "class": pred_class,
        "confidence": confidence,
        "prob_normal": probabilities[0, 0].item(),
        "prob_pathology": probabilities[0, 1].item(),
    }


# =============================================================================
# GradCAM Visualization
# =============================================================================
def generate_gradcam(model, image_tensor: torch.Tensor) -> np.ndarray:
    """Generate GradCAM++ heatmap for model interpretability."""
    if not GRADCAM_AVAILABLE:
        return None

    target_layer = get_last_conv_layer(model)
    if target_layer is None:
        return None

    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])

    image_tensor = image_tensor.to(DEVICE)
    heatmap = cam(input_tensor=image_tensor)

    return heatmap[0]


def overlay_gradcam(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Overlay GradCAM heatmap on the original image."""
    if heatmap is None:
        return None

    H, W = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

    overlay = show_cam_on_image(image, heatmap_resized, use_rgb=True)
    return overlay


# =============================================================================
# Streamlit UI
# =============================================================================
def main():
    st.set_page_config(
        page_title="Chest X-Ray AI Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    st.title("🩺 Chest X-Ray Pathology Detection")
    st.markdown("""
    **Clinical Decision Support Tool** powered by deep learning.

    This application uses an EfficientNetB4 model trained on the MIMIC-CXR dataset
    to detect pathologies in chest X-ray images. The model has been fine-tuned with
    fairness considerations to reduce bias across different clinical subgroups.
    """)

    st.divider()

    # Sidebar - Model Selection & System Info
    with st.sidebar:
        st.header("🤖 Model Selection")

        # Model selector
        selected_model = st.selectbox(
            "Choose a model:",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,
            help="Select the AI model for pathology detection"
        )

        model_config = AVAILABLE_MODELS[selected_model]

        st.caption(model_config["description"])

        metrics = model_config["metrics"]
        cols = st.columns(3)
        cols[0].metric("AUC", f"{metrics['AUC']:.3f}")
        cols[1].metric("Acc", f"{metrics['Accuracy']:.3f}")
        cols[2].metric("F1", f"{metrics['F1']:.3f}")

        st.divider()

        st.header("⚙️ System Information")
        st.info(f"**Device:** {DEVICE}")
        st.info(f"**Input Size:** {IMAGE_SIZE}x{IMAGE_SIZE}")

    # Main Content - Clinical Workflow
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Step 1: Upload X-Ray Image")

        uploaded_file = st.file_uploader(
            "Upload a chest X-ray image (PNG, JPG, DICOM)",
            type=["png", "jpg", "jpeg"],
            help="Upload a frontal chest X-ray for analysis"
        )

        # Demo mode with sample images
        use_demo = st.checkbox("Use demo image (sample from dataset)")

        if use_demo:
            demo_images = []
            demo_dir = "./MIMIC-CXR-png/files-png"
            if os.path.exists(demo_dir):
                for root, dirs, files in os.walk(demo_dir):
                    for f in files:
                        if f.endswith('.png'):
                            demo_images.append(os.path.join(root, f))
                    if len(demo_images) >= 10:
                        break

            if demo_images:
                selected_demo = st.selectbox(
                    "Select a demo image:",
                    demo_images[:10],
                    format_func=lambda x: os.path.basename(x)
                )
                image = Image.open(selected_demo)
                st.image(image, caption="Selected Demo Image", use_container_width=True)
            else:
                st.warning("No demo images found in dataset directory.")
                image = None
        elif uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
        else:
            image = None
            st.info("Upload an image or select from existing.")

    with col2:
        st.header("Step 2: AI Analysis")

        if image is not None:
            # Load model
            model = load_model(selected_model)

            # Preprocess
            with st.spinner("Preprocessing image..."):
                image_tensor = preprocess_image(image)
                display_img = tensor_to_display_image(image_tensor)

            # Run inference
            if st.button("Run Analysis", type="primary", use_container_width=True):
                with st.spinner(f"Running {selected_model} inference..."):
                    results = run_inference(model, image_tensor)

                # Display results
                st.header("Step 3: Results")

                # Model info badge
                st.caption(f"**Model:** {selected_model}")

                # Prediction with color coding
                if results["class"] == 1:
                    st.error(f"**Prediction:** {results['prediction']}")
                else:
                    st.success(f"**Prediction:** {results['prediction']}")

                # Confidence metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        "Normal Probability",
                        f"{results['prob_normal']*100:.1f}%"
                    )
                with col_b:
                    st.metric(
                        "Pathology Probability",
                        f"{results['prob_pathology']*100:.1f}%"
                    )

                # Confidence bar
                st.progress(results["confidence"], text=f"Confidence: {results['confidence']*100:.1f}%")

                # GradCAM visualization
                st.subheader("🔍 Model Interpretability (GradCAM)")

                if GRADCAM_AVAILABLE:
                    with st.spinner("Generating attention map..."):
                        heatmap = generate_gradcam(model, image_tensor)
                        overlay = overlay_gradcam(display_img, heatmap)

                    if overlay is not None:
                        col_orig, col_cam = st.columns(2)
                        with col_orig:
                            st.image(display_img, caption="Preprocessed Image", use_container_width=True)
                        with col_cam:
                            st.image(overlay, caption="GradCAM Attention", use_container_width=True)

                        st.caption("""
                        **GradCAM Interpretation:** The heatmap shows regions the model
                        focused on when making its prediction. Warmer colors (red/yellow)
                        indicate higher attention. This helps clinicians understand the
                        AI's reasoning and verify it aligns with clinical expectations.
                        """)
                    else:
                        st.warning("Could not generate GradCAM visualization.")
                else:
                    st.info("Install `pytorch-grad-cam` for interpretability visualizations.")

                # Clinical disclaimer
                st.divider()
                st.warning("""
                **Clinical Disclaimer:** This tool is for research and educational purposes only.
                It is NOT intended for clinical diagnosis.
                """)
        else:
            st.info("Upload an image to begin analysis.")

if __name__ == "__main__":
    main()
