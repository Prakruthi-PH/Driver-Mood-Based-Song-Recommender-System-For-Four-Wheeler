import os
import random
import urllib.parse
import cv2
from flask import Flask, render_template, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

# Ensure the static folder exists
os.makedirs(os.path.join('static', 'uploaded_images'), exist_ok=True)

# Path for emotion-based songs (folder paths)
EMOTION_SONGS = {
    'happy': 'static/emotion_songs/happy',
    'neutral': 'static/emotion_songs/neutral',
    'angry': 'static/emotion_songs/angry',
    'sad': 'static/emotion_songs/sad',
    'fear': 'static/emotion_songs/fear'
}

# Function to detect emotion using DeepFace
def detect_emotion(image_path):
    try:
        # Use DeepFace to analyze the emotion
        result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=True)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print("Error in emotion detection:", e)
        return 'neutral'  # Default emotion if error occurs

# Function to detect face and process the image with OpenCV
def detect_face(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no faces are detected, return a default emotion and image
    if len(faces) == 0:
        return 'neutral', image_path  # Default emotion if no faces are detected

    # Otherwise, process the first face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around face
        # Crop face for emotion detection
        face_crop = image[y:y+h, x:x+w]
        cv2.imwrite("static/face_crop.jpg", face_crop)  # Save the cropped face image (optional)
        
        # Get emotion for the cropped face
        emotion = detect_emotion("static/face_crop.jpg")
        
        # Add emotion label on the image
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the result image with face rectangles and emotion labels
    result_image_path = 'static/result_image.jpg'
    cv2.imwrite(result_image_path, image)

    return emotion, result_image_path

# Function to randomly select a song from the folder and URL-encode the path
def get_random_song(emotion):
    if emotion in EMOTION_SONGS:
        emotion_folder = EMOTION_SONGS[emotion]
        songs = [f for f in os.listdir(emotion_folder) if f.endswith('.mp3')]  # List all mp3 files
        if songs:
            random_song = random.choice(songs)  # Select a random song
            # Ensure the song path uses forward slashes for URLs and encode it
            song_path = os.path.join(emotion_folder, random_song).replace("\\", "/")
            encoded_song_path = urllib.parse.quote(song_path)  # URL encode the song path
            return encoded_song_path  # Return the encoded song path
    return None  # If no song found for the emotion, return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part'})

    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image to the uploaded_images folder
    image_path = os.path.join('static', 'uploaded_images', image_file.filename)
    image_file.save(image_path)

    # Detect faces and emotions in the uploaded image
    emotion, result_image_path = detect_face(image_path)

    # Get a random song based on the emotion
    song_path = get_random_song(emotion)

    return jsonify({'emotion': emotion, 'result_image': result_image_path, 'song': song_path})

if __name__ == "__main__":
    app.run(debug=True)
