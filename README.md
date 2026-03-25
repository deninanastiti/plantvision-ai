# PlantVision AI

PlantVision AI is a deep learning-based web application for plant image classification.  
This project uses **MobileNetV2** with **transfer learning** and **data augmentation** to classify plant images into multiple plant categories.

## Overview

This application allows users to upload a plant image in JPG, JPEG, or PNG format and receive:
- the predicted plant class  
- the confidence score  
- the top prediction results  

The project was developed as part of a deep learning and computer vision practice project.

## Features

- Plant image classification from uploaded images  
- Confidence score display  
- Top prediction results  
- Streamlit-based web interface  
- MobileNetV2 transfer learning model  

## Model Information

- **Architecture:** MobileNetV2  
- **Task:** Plant Image Classification  
- **Input Size:** 150 x 150  
- **Approach:** Transfer Learning + Data Augmentation  
- **Framework:** TensorFlow / Keras  

## Technologies Used

- Python  
- TensorFlow / Keras  
- Streamlit  
- NumPy  
- Pillow  

## Project Structure

```bash
plantvision-ai/
├── app.py
├── README.md
├── requirements.txt
├── plant_mobilenet_model.keras
└── class_names.json

## How to Run

1. Clone this repository  
2. Open the project folder  
3. Install dependencies  
4. Run the Streamlit application  
5. Open the local URL in your browser  

## Sample Output

The application provides:
- Uploaded image preview
- Predicted plant class
- Confidence score
- Top prediction results

## Notes

This model performs well on clear plant images. However, prediction performance may vary depending on:
- Image quality
- Background complexity
- Visual similarity between plant classes

## Author

**Denina Nastiti Putri Amani**
Deep Learning Project