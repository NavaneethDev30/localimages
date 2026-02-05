import os
import json
from deepface import DeepFace
from PIL import Image
import streamlit as st

# Path to known faces
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Load face info from JSON
INFO_FILE = "face_info.json"
if os.path.exists(INFO_FILE):
    with open(INFO_FILE, "r") as f:
        face_info = json.load(f)
else:
    face_info = {}

# Normalize names for consistent key usage
def normalize_name(name):
    return name.lower().replace(" ", "_")

# Save uploaded image
def save_image(image: Image.Image, path="uploaded.jpg"):
    if image.mode != "RGB":
        image = image.convert("RGB")  # Convert RGBA/other to RGB
    image.save(path)
    return path

# Recognition logic
def recognize_and_describe(image: Image.Image):
    if image is None:
        return "⚠️ Please provide an image."

    try:
        img_path = save_image(image)

        results = DeepFace.find(
            img_path=img_path,
            db_path=KNOWN_FACES_DIR,
            enforce_detection=False
        )

        if len(results) > 0 and len(results[0]) > 0:
            best_match_path = results[0].iloc[0]['identity']
            matched_filename = os.path.basename(best_match_path)
            matched_name = os.path.splitext(matched_filename)[0]
            key = normalize_name(matched_name)
            info = face_info.get(key, "📝 Info not available.")
            return f"✅ Match found: {matched_name.replace('_', ' ').title()}\n\n🧠 Info: {info}"
        else:
            return "❌ No matching face found."

    except Exception as e:
        return f"❌ Error: {str(e)}"

    finally:
        if os.path.exists("uploaded.jpg"):
            os.remove("uploaded.jpg")

# Streamlit UI
st.set_page_config(page_title="🧠 Face Recognition App", layout="centered")
st.title("🧠 Face Recognition App")
st.write("Upload a photo or use your webcam to identify a person and get related information.")

# Image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Live camera capture
captured_image = st.camera_input("Or capture a live image")

# Choose the image to use
image_to_use = uploaded_file or captured_image

if image_to_use is not None:
    image = Image.open(image_to_use)
    st.image(image, caption="Selected Image", use_container_width=True)
    
    if st.button("Recognize Face"):
        result = recognize_and_describe(image)
        st.text(result)
# if __name__ == "__main__":
#     demo.launch(server_name="127.0.0.1", server_port=7860)

