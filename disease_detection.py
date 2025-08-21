import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
import asyncio
import edge_tts
import os
import base64

# ---------------- CONFIG ---------------- #
# Configure Gemini API Key
genai.configure(api_key='AIzaSyDXXHdFN8cYFrZtPIHxpuXQUstOslx1abc')  # replace with your key

# Load your trained models
breast_model = YOLO("breastcancer.pt")   # replace with your model path
brain_model = YOLO("brain_tumor_detector.pt")     # replace with your model path

# ---------------- FUNCTIONS ---------------- #
def run_detection(model, image):
    results = model(image)
    for r in results:
        annotated_img = r.plot()
    return annotated_img, results[0].boxes

def generate_description(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

async def text_to_speech(text, lang="en-US-AriaNeural"):
    communicate = edge_tts.Communicate(text, voice=lang)
    output_file = "output.mp3"
    await communicate.save(output_file)
    return output_file

def play_audio(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/mp3")

# ---------------- STREAMLIT UI ---------------- #
st.title("🩺 Disease Detection with Voice Description")

choice = st.radio("Select Detection Model:", ["Breast Cancer", "Brain Tumor"])
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

lang_choice = st.selectbox(
    "Select Voice Language:",
    ["en-US-AriaNeural", "en-GB-RyanNeural", "hi-IN-MadhurNeural", "te-IN-ShrutiNeural", "ta-IN-PallaviNeural"]
)

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)

    if choice == "Breast Cancer":
        result_img, boxes = run_detection(breast_model, img)
        prompt = "Explain the detection result of breast cancer in medical terms for patient understanding."
    else:
        result_img, boxes = run_detection(brain_model, img)
        prompt = "Explain the detection result of brain tumor in medical terms for patient understanding."

    # Show detection result
    st.image(result_img, channels="BGR", caption="Detection Result", use_container_width=True)

    # Generate description
    description = generate_description(prompt)
    st.subheader("📖 AI Explanation:")
    st.write(description)

    # Generate TTS voice
    audio_path = asyncio.run(text_to_speech(description, lang_choice))
    play_audio(audio_path)

