from flask import Flask, request, jsonify, render_template
from moviepy import VideoFileClip
import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
FRAME_FOLDER = "frames"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

print("Loading AI models...")

# Audio model
whisper_model = whisper.load_model("tiny")

# Semantic similarity model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Image caption model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Object detection model
yolo_model = YOLO("yolov8n.pt")

print("Models loaded successfully")


@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------
# Extract audio
# ---------------------------
def extract_audio(video_path):

    print("Extracting audio")

    audio_path = video_path.replace(".mp4", ".wav")

    video = VideoFileClip(video_path)

    video.audio.write_audiofile(audio_path)

    return audio_path


# ---------------------------
# Speech to text
# ---------------------------
def speech_to_text(audio_path):

    print("Running Whisper")

    result = whisper_model.transcribe(audio_path)

    return result["text"]


# ---------------------------
# Extract frames
# ---------------------------
def extract_frames(video_path):

    print("Extracting frames")

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    interval = fps * 3

    frames = []

    count = 0
    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if count % interval == 0:

            frame_path = f"{FRAME_FOLDER}/frame{frame_id}.jpg"

            cv2.imwrite(frame_path, frame)

            frames.append(frame_path)

            frame_id += 1

        count += 1

    cap.release()

    return frames


# ---------------------------
# YOLO Object Detection
# ---------------------------
def detect_objects(frame_path):

    results = yolo_model(frame_path)

    detected_objects = []

    for r in results:
        for box in r.boxes:
            label = yolo_model.names[int(box.cls)]
            detected_objects.append(label)

    return list(set(detected_objects))


# ---------------------------
# Day/Night Detection
# ---------------------------
def detect_time(frame_path):

    img = cv2.imread(frame_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)

    if brightness > 100:
        return "daytime"
    else:
        return "nighttime"


# ---------------------------
# Convert detections to English sentence
# ---------------------------
def generate_semantic_paragraph(objects, time_of_day):

    sentence = ""

    if "person" in objects:
        sentence += "A person is visible in the scene. "

    animals = ["dog", "cat", "cow", "horse", "bird"]

    detected_animals = [a for a in animals if a in objects]

    if detected_animals:
        sentence += "An animal such as " + ", ".join(detected_animals) + " appears in the video. "

    if "ball" in objects:
        sentence += "Someone appears to be interacting with a ball. "

    sentence += f"The video appears to be recorded during the {time_of_day}. "

    return sentence


# ---------------------------
# Describe frames
# ---------------------------
def describe_frames(frame_paths):

    descriptions = []

    for frame in frame_paths:

        image = Image.open(frame).convert("RGB")

        inputs = processor(image, return_tensors="pt")

        output = blip_model.generate(**inputs)

        caption = processor.decode(output[0], skip_special_tokens=True)

        objects = detect_objects(frame)

        time_of_day = detect_time(frame)

        semantic_sentence = generate_semantic_paragraph(objects, time_of_day)

        final_description = caption + " " + semantic_sentence

        descriptions.append(final_description)

    return " ".join(descriptions)


# ---------------------------
# Semantic similarity
# ---------------------------
def similarity_score(reference, combined_text):

    ref_vec = embedding_model.encode(reference)

    video_vec = embedding_model.encode(combined_text)

    score = cosine_similarity([ref_vec], [video_vec])[0][0]

    return float(round(score * 100, 2))


# ---------------------------
# Main analysis
# ---------------------------
@app.route("/analyze", methods=["POST"])
def analyze():

    video = request.files["video"]
    reference = request.form["reference"]

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)

    video.save(video_path)

    print("Video received")

    audio_path = extract_audio(video_path)

    transcript = speech_to_text(audio_path)

    print("Transcript:", transcript)

    frame_paths = extract_frames(video_path)

    frame_description = describe_frames(frame_paths)

    print("Frame description:", frame_description)

    combined_text = transcript + " " + frame_description

    score = similarity_score(reference, combined_text)

    return jsonify({
        "transcript": transcript,
        "frame_description": frame_description,
        "score": score
    })


if __name__ == "__main__":
    app.run(debug=True)
