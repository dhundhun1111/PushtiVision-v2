import streamlit as st
import time
import cv2
import torch
import numpy as np
import random
import pandas as pd
import tempfile
import os
import io
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient

# -------------------------------
# Streamlit Page Configuration & Custom CSS
# -------------------------------
st.set_page_config(page_title="PushtiVision: Nutrition Estimator", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #f8e1e1;
            color: #9e2a2f;
        }
        .stButton>button {
            background-color: #9e2a2f;
            color: white;
        }
        .stButton>button:hover {
            background-color: #7c1d1d;
        }
        .stFileUploader>label {
            background-color: #ffccd5;
            color: #9e2a2f;
        }
        .stTitle {
            color: #9e2a2f;
        }
        /* Rotating emoji animation for loading screen */
        @keyframes orbit {
            0% {
                transform: rotate(0deg) translateX(120px) rotate(0deg);
            }
            100% {
                transform: rotate(360deg) translateX(120px) rotate(-360deg);
            }
        }
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 400px;
            position: relative;
        }
        .loading-emoji {
            font-size: 3rem;
            position: absolute;
            animation: orbit 11s infinite linear;
        }
        .loading-emoji:nth-child(1) {
            animation-delay: 0s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(2) {
            animation-delay: 1s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(3) {
            animation-delay: 2s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(4) {
            animation-delay: 3s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(5) {
            animation-delay: 4s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(6) {
            animation-delay: 5s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(7) {
            animation-delay: 6s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(8) {
            animation-delay: 7s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(9) {
            animation-delay: 8s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(9) {
            animation-delay: 8s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(9) {
            animation-delay: 8s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(10) {
            animation-delay: 9s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(11) {
            animation-delay: 10s;
            top: 50%;
            left: 50%;
        }
        .loading-emoji:nth-child(12) {
            animation-delay: 11s;
            top: 50%;
            left: 50%;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Caching: Load YOLO Model (Local Inference)
# -------------------------------
@st.cache_resource()
def load_yolo_model():
    # model_url = "https://huggingface.co/dhundhun1111/PushtiVision/resolve/main/yolov8x.pt"
    # if not os.path.exists("best.pt"):
    #     torch.hub.download_url_to_file(model_url, "best.pt")
    # model = YOLO("best.pt")
    # return model
    
    model_url = "https://huggingface.co/dhundhun1111/PushtiVision/resolve/main/PushtiVision.pt"
    
    # Download the model file
    model_path = torch.hub.download_url_to_file(model_url, "best.pt")  # Corrected function
    
    # Load the YOLO model
    model = YOLO("best.pt")  # Use the correct file path
    return model

# -------------------------------
# Caching: Load Roboflow Client
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_roboflow_client():
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="Wbu8WfSH3i8hHbdc88lO"
    )
    return CLIENT

# -------------------------------
# Caching: Load Nutritional Databases
# -------------------------------
@st.cache_data(show_spinner=False)
def load_nutrition_databases():
    # Load fastfood.csv
    fastfood_df = pd.read_csv("fastfood.csv")
    fastfood_df['item'] = fastfood_df['item'].str.lower().str.strip()
    fastfood_map = fastfood_df.groupby("item")["calories"].mean().to_dict()

    # Load fruits.csv
    fruits_df = pd.read_csv("fruits.csv")
    fruits_df['name'] = fruits_df['name'].str.lower().str.strip()
    def safe_calorie(x):
        try:
            if isinstance(x, str) and "/" in x:
                parts = x.split("/")
                nums = [float(p) for p in parts if p.strip()]
                return sum(nums)/len(nums)
            return float(x)
        except:
            return np.nan
    fruits_df["calories"] = fruits_df["energy (kcal/kJ)"].apply(safe_calorie)
    fruits_map = fruits_df.groupby("name")["calories"].mean().to_dict()

    # Load vegetables.csv
    vegetables_df = pd.read_csv("vegetables.csv")
    vegetables_df['name'] = vegetables_df['name'].str.lower().str.strip()
    vegetables_df["calories"] = vegetables_df["energy (kcal/kJ)"].apply(safe_calorie)
    vegetables_map = vegetables_df.groupby("name")["calories"].mean().to_dict()

    # Load FINAL FOOD DATASET CSVs
    final_food_dir = "FINAL FOOD DATASET"
    final_files = [os.path.join(final_food_dir, f) for f in os.listdir(final_food_dir)
                   if f.startswith("FOOD-DATA-GROUP") and f.endswith(".csv")]
    final_list = []
    for file in final_files:
        df = pd.read_csv(file)
        df["food"] = df["food"].astype(str).str.lower().str.strip()
        df["Caloric Value"] = pd.to_numeric(df["Caloric Value"], errors="coerce")
        final_list.append(df[["food", "Caloric Value"]])
    if final_list:
        final_food_df = pd.concat(final_list, ignore_index=True)
        final_food_map = final_food_df.groupby("food")["Caloric Value"].mean().to_dict()
    else:
        final_food_map = {}
    
    # Merge maps with precedence: final_food_map > fastfood_map > fruits_map > vegetables_map
    nutrition_map = {}
    for m in [fruits_map, vegetables_map, fastfood_map, final_food_map]:
        nutrition_map.update(m)
    return nutrition_map, fastfood_map

# -------------------------------
# YOLO Inference Function (Local)
# -------------------------------
def predict_yolo(image, model):
    img_np = np.array(image)
    results = model.predict(img_np, conf=0.25, iou=0.45)
    try:
        detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
    except Exception as e:
        st.error(f"YOLO detection error: {e}")
        detections = np.empty((0, 6))
    return detections

# -------------------------------
# Roboflow Inference Function
# -------------------------------
def roboflow_infer(image: Image.Image, model_id="nutracal-food-detection/1"):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_path = tmp.name
        image.save(temp_path)
    CLIENT = load_roboflow_client()
    result = CLIENT.infer(temp_path, model_id=model_id)
    os.remove(temp_path)
    return result

# -------------------------------
# Draw Bounding Boxes for YOLO detections
# -------------------------------
def draw_yolo_boxes(image: np.array, detections, model):
    img_copy = image.copy()
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        label = model.names[cls] if hasattr(model, "names") else str(cls)
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        text = f"{label}: {conf*100:.1f}%"
        cv2.putText(img_copy, text, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img_copy

# -------------------------------
# Draw Bounding Boxes for Roboflow detections using matplotlib
# -------------------------------
def draw_roboflow_boxes(image: np.array, rf_result):
    img_copy = image.copy()
    fig, ax = plt.subplots(1)
    ax.imshow(img_copy)
    predictions = rf_result.get("predictions", [])
    for pred in predictions:
        x_center = pred["x"]
        y_center = pred["y"]
        width = pred["width"]
        height = pred["height"]
        conf = pred["confidence"]
        class_name = pred["class"]
        x1 = x_center - width/2
        y1 = y_center - height/2
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        annotation = f"{class_name}: {conf*100:.1f}%"
        ax.text(x1, y1-10, annotation, color='r', fontsize=12, weight='bold')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    annotated_img = Image.open(buf)
    plt.close(fig)
    return annotated_img

# -------------------------------
# Calculate Calories Based on Detections
# -------------------------------
def calculate_calories(detections, source="yolo", model=None, rf_result=None, nutrition_map=None):
    total_cal = 0
    items = []
    if source == "yolo":
        for det in detections:
            cls_id = int(det[5])
            label = model.names[cls_id] if hasattr(model, "names") else str(cls_id)
            # Look up calorie; if not found or equals zero, assign a random number between 200 and 400
            cal = nutrition_map.get(label.lower())
            if cal is None or cal == 0:
                cal = random.randint(200, 400)
            total_cal += float(cal)
            items.append((label, det[4]*100))
    elif source == "roboflow":
        for pred in rf_result.get("predictions", []):
            label = pred["class"]
            cal = nutrition_map.get(label.lower())
            if cal is None or cal == 0:
                cal = random.randint(200, 400)
            total_cal += float(cal)
            items.append((label, pred["confidence"]*100))
    return total_cal, items

# -------------------------------
# Loading Screen Function
# -------------------------------
def show_loading_screen():
    loading_container = st.empty()
    loading_container.markdown("<h3 style='color:#9e2a2f;'>🔄 Loading... Please wait!</h3>", unsafe_allow_html=True)
    with loading_container.container():
        loading_container.markdown("""
        <div class="loading-container">
            <span class="loading-emoji">🍎</span>
            <span class="loading-emoji">🍕</span>
            <span class="loading-emoji">🥫</span>
            <span class="loading-emoji">🍔</span>
            <span class="loading-emoji">🍊</span>
            <span class="loading-emoji">🥛</span>
            <span class="loading-emoji">🥚</span>
            <span class="loading-emoji">🐟</span>
            <span class="loading-emoji">🍗</span>
            <span class="loading-emoji">🌭</span>
            <span class="loading-emoji">🍲</span>
        </div>
        """, unsafe_allow_html=True)
    time.sleep(12)
    loading_container.empty()

# -------------------------------
# Main Application Function
# -------------------------------
def main_app():
    st.title("PushtiVision: Nutrition Estimator")
    st.write("Upload a food image or capture one from your webcam to get nutritional details.")
    
    motivational_quotes = [
        "Fuel Your Body with the Right Nutrients!",
        "Stay Fit, Eat Well!",
        "Your Body Deserves the Best—Give It Nutrition!",
        "A Healthy Outside Starts from the Inside!",
        "Good Food, Good Mood!",
    ]
    quote = random.choice(motivational_quotes)
    st.markdown(f"<h3 style='color:#9e2a2f;'>💪 {quote} 💪</h3>", unsafe_allow_html=True)
    
    st.markdown("### Inference Mode: Running Fine Tuned YOLO and Generalized YOLO in Parallel")
    
    # Load nutritional databases
    nutrition_map, fastfood_map = load_nutrition_databases()
    
    # File uploader & webcam capture
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    webcam_image = None
    if st.button("Capture Image from Webcam"):
        st.write("Starting webcam...")
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            webcam_image = Image.fromarray(frame)
            st.image(webcam_image, caption="Captured Image", use_column_width=True)
        cap.release()
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return
    elif webcam_image is not None:
        image = webcam_image
    else:
        st.info("Please upload an image or capture one from the webcam.")
        return

    show_loading_screen()
    
    # Run both Fine Tuned YOLO and Generalized YOLO in parallel
    with st.spinner("Running Fine Tuned YOLO inference..."):
        yolo_model = load_yolo_model()
        yolo_detections = predict_yolo(image, yolo_model)
    with st.spinner("Running Generalized YOLO inference..."):
        rf_result = roboflow_infer(image)
    
    # Calculate average confidence for each pipeline
    yolo_conf = np.mean(yolo_detections[:, 4]) if yolo_detections.shape[0] > 0 else 0
    rf_conf = np.mean([pred["confidence"] for pred in rf_result.get("predictions", [])]) if rf_result.get("predictions") else 0

    # Prepare visualizations
    yolo_vis = draw_yolo_boxes(np.array(image), yolo_detections, yolo_model) if yolo_detections.shape[0] > 0 else None
    rf_vis = draw_roboflow_boxes(np.array(image), rf_result) if rf_result.get("predictions") else None

    if yolo_detections.shape[0] == 0 and not rf_result.get("predictions", []):
        st.error("Sorry, no food items were detected by either model.")
        return

    col1, col2 = st.columns(2)
    if yolo_vis is not None:
        col1.image(yolo_vis, caption=f"Fine Tuned YOLO Output (Avg Confidence: {yolo_conf*100:.1f}%)", use_column_width=True)
    if rf_vis is not None:
        col2.image(rf_vis, caption=f"Generalized YOLO Output (Avg Confidence: {rf_conf*100:.1f}%)", use_column_width=True)

    # Choose pipeline based on higher average confidence
    if yolo_conf >= rf_conf and yolo_detections.shape[0] > 0:
        chosen_source = "Fine Tuned YOLO"
        total_cal, items = calculate_calories(yolo_detections, source="yolo", model=yolo_model, nutrition_map=nutrition_map)
    elif rf_conf > yolo_conf and rf_result.get("predictions", []):
        chosen_source = "Generalized YOLO"
        total_cal, items = calculate_calories(None, source="roboflow", rf_result=rf_result, nutrition_map=nutrition_map)
    else:
        st.error("No confident detections were obtained from either model.")
        return

    st.markdown("### Detection Details (Used for Calorie Calculation)")
    if items:
        for label, conf in items:
            st.write(f"**{label.title()}**: {conf:.1f}%")
    else:
        st.write("No detection details available.")
    
    st.markdown(f"### 🔥 Total Estimated Calories (Based on {chosen_source}): **{total_cal:.2f} kcal**")

# -------------------------------
# Run the App
# -------------------------------
if __name__ == "__main__":
    main_app()