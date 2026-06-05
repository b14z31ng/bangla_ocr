import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms
import pandas as pd
import plotly.express as px


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "cnn_model.pth"
LABELS_PATH = ROOT_DIR / "labels.json"
MODEL_INFO_PATH = ROOT_DIR / "model_info.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MAP
# ============================================================

CHARACTER_MAP_PATH = ROOT_DIR / "charechter_map.json"

@st.cache_data
def load_character_map():
    with open(CHARACTER_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}

    for item in data["bengali_character_sheet"]:
        mapping[str(item["id"])] = item["character"]

    return mapping

character_map = load_character_map()


# ============================================================
# MODEL
# ============================================================

class BanglaCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    with open(MODEL_INFO_PATH, "r") as f:
        model_info = json.load(f)

    num_classes = model_info["num_classes"]

    model = BanglaCNN(num_classes=num_classes)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
labels_map = load_labels()


# ============================================================
# IMAGE TRANSFORM
# ============================================================

inference_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# ============================================================
# PREDICTION
# ============================================================

def predict_character(pil_image):
    tensor = inference_transform(pil_image)
    tensor = tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(probs, k=5)

    top_probs = top_probs[0].cpu().numpy()
    top_indices = top_indices[0].cpu().numpy()

    pred_idx = int(top_indices[0])
    confidence = float(top_probs[0])

    predicted_class = labels_map[str(pred_idx)]

    predicted_character = character_map.get(
        predicted_class,
        predicted_class
    )

    top_predictions = []

    for idx, prob in zip(top_indices, top_probs):

        folder_id = labels_map[str(int(idx))]

        char = character_map.get(
            folder_id,
            folder_id
        )

        top_predictions.append({
            "character": char,
            "folder": folder_id,
            "probability": float(prob)
        })

    return (
        predicted_character,
        predicted_class,
        confidence,
        top_predictions
    )

# ============================================================
# SEGMENTATION
# ============================================================

def segment_characters(canvas_image):
    """
    Segment handwritten Bangla word into individual characters using a projection-based approach.
    Falls back to contour-based segmentation if the primary method fails.
    """
    # --- Image Preprocessing ---
    image = cv2.cvtColor(canvas_image, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # --- Primary Segmentation: Projection-based ---
    try:
        # --- Matra (Headline) Removal ---
        # 1. Horizontal projection to find the matra
        horizontal_projection = np.sum(thresh, axis=1)

        # Find the row with the maximum projection (likely the matra)
        matra_row = np.argmax(horizontal_projection)

        # Create a copy of the thresholded image to work on
        no_matra_image = thresh.copy()

        # 2. Remove the matra by setting its pixels to 0
        # This helps in separating characters connected by the headline.
        # We remove a few rows around the peak to ensure complete removal.
        no_matra_image[matra_row-2:matra_row+2, :] = 0


        # --- Character Segmentation using Vertical Projection ---
        # 3. Vertical projection on the matra-less image
        vertical_projection = np.sum(no_matra_image, axis=0)

        # 4. Find gaps between characters
        # Gaps are where the vertical projection is zero (or close to zero)
        gaps = np.where(vertical_projection == 0)[0]

        # 5. Determine character boundaries from the gaps
        boundaries = []
        start = 0
        for i in range(1, len(gaps)):
            # If a gap is significant (more than a few pixels wide), it's a boundary
            if gaps[i] - gaps[i-1] > 2:
                end = gaps[i-1]
                # Find the actual start of the character by looking for the first non-zero projection
                char_pixels = np.where(vertical_projection[start:end] > 0)[0]
                if len(char_pixels) > 0:
                    true_start = start + char_pixels[0]
                    boundaries.append((true_start, end))
                start = gaps[i]
        
        # Add the last character boundary
        char_pixels = np.where(vertical_projection[start:] > 0)[0]
        if len(char_pixels) > 0:
            true_start = start + char_pixels[0]
            boundaries.append((true_start, len(vertical_projection)))


        # --- Crop and Refine Characters ---
        boxes = []
        for start, end in boundaries:
            # Crop from the original thresholded image (with matra)
            char_crop = thresh[:, start:end]
            
            # Use findContours on the small crop to get a tight bounding box
            contours, _ = cv2.findContours(char_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue

            # Combine all contours in the crop to get a single bounding box
            all_points = np.concatenate([c for c in contours])
            x, y, w, h = cv2.boundingRect(all_points)

            # Filter out very small noise boxes
            if w * h < 100:
                continue
            
            # Adjust box coordinates to be relative to the full image
            boxes.append((x + start, y, w, h))

        # Sort boxes left-to-right
        boxes = sorted(boxes, key=lambda b: b[0])
        
        # If projection method fails, len(boxes) will be 0 or 1.
        # The 'if' condition in the fallback logic will catch this.

    except Exception:
        # If any error occurs in the projection method, clear boxes to trigger fallback.
        boxes = []


    # --- Fallback Segmentation: Contour-based ---
    # If the projection method results in 1 or 0 characters, it likely failed.
    # This can happen with isolated characters or unusual writing styles.
    # In such cases, we fall back to the simpler contour-based method.
    if len(boxes) <= 1:
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fallback_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 100:  # Area check
                fallback_boxes.append((x, y, w, h))
        
        # Only use fallback if it provides a better result
        if len(fallback_boxes) > len(boxes):
            boxes = sorted(fallback_boxes, key=lambda b: b[0])


    # --- Final Character Extraction and Debug Image Generation ---
    characters = []
    debug_image = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)

    for x, y, w, h in boxes:
        # Draw bounding box on the debug image
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add padding to the character crop
        padding = 10
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(thresh.shape[1], x + w + padding)
        y2 = min(thresh.shape[0], y + h + padding)

        # Crop from the original binary image
        crop = thresh[y1:y2, x1:x2]

        # Convert to PIL Image for the model
        pil_crop = Image.fromarray(crop)

        characters.append({
            "image": pil_crop,
            "bbox": (x, y, w, h)
        })

    return characters, thresh, debug_image


# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title="Bangla OCR",
    layout="wide"
)

st.title("Bangla Handwritten OCR")
st.markdown(
    """
Draw a Bangla word below.

The system will:

1. Segment characters
2. Classify each character
3. Reconstruct the predicted sequence
"""
)

canvas_result = st_canvas(
    fill_color="rgba(255,255,255,0)",
    stroke_width=12,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=800,
    height=250,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:

    if st.button("Predict"):

        canvas_image = canvas_result.image_data.astype(np.uint8)

        characters, binary_image, debug_image = segment_characters(
            canvas_image
        )

        st.subheader("Binary Image")

        st.image(
            binary_image,
            use_container_width=True
        )

        st.subheader("Segmentation Debug")

        st.image(
            debug_image,
            use_container_width=True
        )

        st.info(f"Detected Segments: {len(characters)}")

        st.subheader("Binary Image")

        st.image(
            binary_image,
            use_container_width=True
        )

        if len(characters) == 0:
            st.warning(
                "No characters detected. Please draw a word."
            )
            st.stop()

        predictions = []

        st.subheader("Segmented Characters")

        cols = st.columns(min(len(characters), 6))

        for idx, character in enumerate(characters):

            char_img = character["image"]

            predicted_character, predicted_class, confidence, top_predictions = predict_character(
    char_img
)
            predictions.append(predicted_character)

            with cols[idx % len(cols)]:
                st.image(
                    char_img,
                    caption=f"{predicted_character} ({predicted_class})\n{confidence:.2%}",
                    use_container_width=True
                )
            with st.expander(
                f"Top 5 predictions for character {idx+1}",
                expanded=False
            ):

                df = pd.DataFrame({
                    "Character": [
                        p["character"]
                        for p in top_predictions
                    ],
                    "Probability": [
                        p["probability"] * 100
                        for p in top_predictions
                    ]
                })

                fig = px.bar(
                    df,
                    x="Character",
                    y="Probability",
                    text="Probability",
                    title="Top 5 Character Predictions"
                )

                fig.update_traces(
                    texttemplate="%{y:.1f}%",
                    textposition="outside"
                )

                fig.update_layout(
                    yaxis_title="Probability (%)",
                    xaxis_title="Character",
                    height=350
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                for pred in top_predictions:
                    st.write(
                        f"{pred['character']} ({pred['folder']}) → {pred['probability']:.2%}"
                    )
        reconstructed_word = "".join(predictions)

        st.subheader("Predicted Sequence")

        st.success(reconstructed_word)

        st.subheader("Detailed Predictions")

        for i, pred in enumerate(predictions):
            st.write(
                f"Character {i+1}: {pred}"
            )

st.sidebar.title("Model Information")

try:
    with open(MODEL_INFO_PATH, "r") as f:
        info = json.load(f)

    st.sidebar.json(info)

except Exception:
    st.sidebar.warning(
        "Could not load model information."
    )

st.sidebar.write(f"Device: {DEVICE}")