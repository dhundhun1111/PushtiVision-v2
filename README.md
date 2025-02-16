# PushtiVision: The Food Nutrition Estimator from Food Images

PushtiVision is a deep learning-based application designed to detect food items from images and estimate their nutritional content. The project leverages state-of-the-art models such as YOLOv5 for object detection and VGG19 for additional feature extraction. It is built upon a custom dataset sourced and processed via the Roboflow API, and it has been fine-tuned through multiple training cycles to optimize performance, particularly for Indian food items.

---
## YOLOv5-Based Object Detection Pipeline
![yolo drawio](https://github.com/user-attachments/assets/8b84e70e-d798-42d1-9c26-b7de110260ff)



## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Tech Stack](#tech-stack)
- [Pipeline Overview](#pipeline-overview)
  - [1. Dataset Preparation](#1-dataset-preparation)
  - [2. Object Detection Using YOLOv5](#2-object-detection-using-yolov5)
  - [3. Feature Extraction Using VGG19](#3-feature-extraction-using-vgg19)
  - [4. Deployment](#4-deployment)
- [Environment & Dependencies](#environment--dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
  - [Streamlit Web App](#streamlit-web-app)
- [Enhancements & Future Work](#enhancements--future-work)
- [Contributors](#contributors)

---

## Overview

The Food Nutrition Estimator is designed to analyze food images and provide detailed nutritional insights. The core functionality involves:

- *Object Detection:* Using YOLOv5 to identify food items in an image.
- *Feature Extraction:* Leveraging a pre-trained VGG19 network (fine-tuned on our dataset) to further refine the classification of detected items.
- *Nutritional Estimation:* Mapping the detected food items to their corresponding calorie and macronutrient values using a dedicated nutritional database.
- *Deployment:* Providing a user-friendly web interface via Streamlit for real-time inference.

---

## Motivation

Accurate calorie and nutrient estimation is crucial for maintaining a healthy diet. With the growing awareness of personalized nutrition, PushtiVision assists users in tracking their food intake and understanding the nutritional composition of their meals. Specifically, this project addresses the need for localized nutritional estimation by focusing on Indian food items.

---

## Tech Stack

- *Deep Learning Frameworks:* PyTorch, TensorFlow
- *Object Detection:* YOLOv5 (Transfer Learning with YOLOv5s)
- *Feature Extraction:* VGG19 (pre-trained and fine-tuned)
- *Dataset Management:* Roboflow API
- *Programming Language:* Python
- *Deployment:* Streamlit (Web UI)
- *Environment:* Google Colab (Training), Docker (Packaging and Deployment)

---

## Pipeline Overview

### 1. Dataset Preparation

- *Data Sourcing & Annotation:*  
  - The dataset is sourced using the [Roboflow API](https://roboflow.com/) and consists of high-quality images of various food items.
  - Images are annotated for food item classification.
  
- *Data Augmentation:*  
  - Techniques such as flipping, scaling, and HSV (Hue, Saturation, Value) transformations are applied.
  - Custom augmentation (e.g., mosaic, mixup) is integrated to improve model robustness.

- *YOLO Format:*  
  - The images and their corresponding annotations are formatted to meet YOLO requirements (i.e., images alongside text files containing bounding box coordinates).

### 2. Object Detection Using YOLOv5

- *Model Configuration:*  
  - *Image Size:* 640x640
  - *Batch Size:* 16
  - *Epochs:* 100+ with fine-tuning
  - *Pretrained Weights:* Transfer learning from YOLOv5s
- *Hyperparameter Tuning:*  
  - Evolution-based optimization techniques are used to fine-tune the model’s hyperparameters.
- *Training Environment:*  
  - The training is conducted on Google Colab with GPU acceleration.

### 3. Feature Extraction Using VGG19

- *Purpose:*  
  - VGG19 is used to further refine food classification and extract additional feature representations.
- *Fine-Tuning:*  
  - A pre-trained VGG19 model is fine-tuned on the custom dataset.
  - The model is later converted to TorchScript for efficient inference.

### 4. Deployment

- *Web-Based UI:*  
  - The application is deployed using [Streamlit](https://streamlit.io/).
  - Users can upload food images or capture images via a webcam.
- *Inference Pipeline:*  
  1. *Detection:* YOLOv5 detects food items in an image.
  2. *Feature Extraction:* The detected regions are processed through VGG19.
  3. *Nutritional Mapping:* Detected items are matched with a nutritional database that provides calorie and macronutrient information.
- *Packaging:*  
  - Docker is used for packaging the application, ensuring consistency across different environments.

---

## Environment & Dependencies

### Dependencies

- *Python:* 3.8+
- *PyTorch:* For deep learning model implementation.
- *TensorFlow:* For potential additional model support.
- *YOLOv5:* Object detection framework (available from [ultralytics/yolov5](https://github.com/ultralytics/yolov5)).
- *VGG19:* Pre-trained network available via torchvision or TensorFlow.
- *Roboflow API:* For dataset sourcing and annotation.
- *Streamlit:* For the web-based user interface.
- *Docker:* For containerization.
- *Additional Python Packages:*
  - opencv-python
  - numpy
  - Pillow
  - time
  - random
  - (Other dependencies as per requirements in requirements.txt)

### Setting Up the Environment

1. *Clone the Repository:*

   ```bash
   git clone https://github.com/dhundhun1111/PushtiVision-v2.git
   cd PushtiVision-v2
