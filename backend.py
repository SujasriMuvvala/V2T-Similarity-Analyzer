from flask import Flask, request, jsonify, render_template
from moviepy import VideoFileClip
import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
FRAME_FOLDER = "frames"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

print("Loading AI models...")

# audio model
whisper_model = whisper.load_model("tiny")

# semantic model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# image caption model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

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

    transcript = result["text"]

    return transcript


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
# Describe frames
# ---------------------------
def describe_frames(frame_paths):

    descriptions = []

    for frame in frame_paths:

        image = Image.open(frame).convert("RGB")

        inputs = processor(image, return_tensors="pt")

        output = blip_model.generate(**inputs)

        caption = processor.decode(output[0], skip_special_tokens=True)

        descriptions.append(caption)

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
    # ================================
    # IMPORT LIBRARIES
    # ================================

    import cv2
    import numpy as np
    from ultralytics import YOLO

    # ================================
    # LOAD YOLO MODEL (runs once)
    # ================================

    # YOLOv8 model for detecting objects like person, dog, ball etc.
    yolo_model = YOLO("yolov8n.pt")


    # ================================
    # FUNCTION 1: OBJECT DETECTION
    # ================================

    def detect_objects(frame_path):
        """
        Detect objects in the frame using YOLO
        Returns a list of detected objects
        """

        results = yolo_model(frame_path)

        detected_objects = []

        for r in results:
            for box in r.boxes:
                # get object label name
                label = yolo_model.names[int(box.cls)]

                detected_objects.append(label)

        # remove duplicate detections
        return list(set(detected_objects))


    # ================================
    # FUNCTION 2: DAY / NIGHT DETECTION
    # ================================

    def detect_time(frame_path):
        """
        Detect whether the scene is daytime or nighttime
        using brightness of the image
        """

        img = cv2.imread(frame_path)

        # convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # calculate brightness
        brightness = np.mean(gray)

        if brightness > 100:
            return "daytime"
        else:
            return "nighttime"


    # ================================
    # FUNCTION 3: GENERATE ENGLISH SENTENCE
    # ================================

    def generate_semantic_paragraph(objects, time_of_day):
        """
        Convert detected objects into natural English sentences
        """

        sentence = ""

        # detect person
        if "person" in objects:
            sentence += "A person is visible in the scene. "

        # detect animals
        animals = ["dog", "cat", "cow", "horse", "bird"]

        detected_animals = [animal for animal in animals if animal in objects]

        if detected_animals:
            sentence += "An animal such as " + ", ".join(detected_animals) + " appears in the video. "

        # detect ball activity
        if "ball" in objects:
            sentence += "Someone seems to be interacting with a ball. "

        # add time description
        sentence += f"The video appears to be recorded during the {time_of_day}. "

        return sentence


    # ================================
    # FUNCTION 4: MAIN VISUAL SEMANTIC GENERATION
    # ================================

    def generate_visual_semantics(frame_path, caption):
        """
        Combine BLIP caption with object detection and
        generate a semantic paragraph
        """

        # detect objects
        objects = detect_objects(frame_path)

        # detect day or night
        time_of_day = detect_time(frame_path)

        # convert detections into english sentence
        semantic_sentence = generate_semantic_paragraph(objects, time_of_day)

        # combine with caption generated by BLIP
        final_description = caption + " " + semantic_sentence

        return final_description