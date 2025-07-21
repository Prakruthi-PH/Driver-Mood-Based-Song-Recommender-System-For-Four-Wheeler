# # import os
# # from flask import Flask, render_template, request, jsonify
# # from deepface import DeepFace

# # app = Flask(__name__)

# # # Ensure the static folder exists
# # os.makedirs(os.path.join('static', 'uploaded_images'), exist_ok=True)

# # # Function to detect emotion using DeepFace
# # def detect_emotion(image_path):
# #     try:
# #         # Use DeepFace to analyze the emotion
# #         result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
# #         emotion = result[0]['dominant_emotion']
# #         return emotion
# #     except Exception as e:
# #         print("Error in emotion detection:", e)
# #         return None

# # # Function to detect face and process the image with OpenCV
# # def detect_face(image_path):
# #     # Load the image
# #     image = cv2.imread(image_path)
    
# #     # Convert the image to grayscale for face detection
# #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
# #     # Load pre-trained Haar Cascade for face detection
# #     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# #     # Detect faces in the image
# #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# #     # Draw rectangles around faces and annotate the emotion
# #     for (x, y, w, h) in faces:
# #         cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around face
# #         # Crop face for emotion detection
# #         face_crop = image[y:y+h, x:x+w]
# #         cv2.imwrite("static/face_crop.jpg", face_crop)  # Save the cropped face image (optional)
# #         # Get emotion for the cropped face
# #         emotion = detect_emotion("static/face_crop.jpg")
# #         # Add emotion label on the image
# #         cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# #     # Save the result image with face rectangles and emotion labels
# #     result_image_path = 'static/result_image.jpg'
# #     cv2.imwrite(result_image_path, image)

# #     return result_image_path

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/upload', methods=['POST'])
# # def upload():
# #     if 'image' not in request.files:
# #         return jsonify({'error': 'No image file part'})

# #     image_file = request.files['image']
    
# #     if image_file.filename == '':
# #         return jsonify({'error': 'No selected file'})

# #     # Save the uploaded image to the uploaded_images folder
# #     image_path = os.path.join('static', 'uploaded_images', image_file.filename)
# #     image_file.save(image_path)

# #     # Detect faces and emotions in the uploaded image
# #     emotion = detect_emotion(image_path)

# #     # Return the emotion as a JSON response
# #     return jsonify({'emotion': emotion})

# # if __name__ == "__main__":
# #     app.run(debug=True)

# import os
# import cv2
# from flask import Flask, render_template, request, jsonify
# from deepface import DeepFace

# app = Flask(__name__)

# # Ensure the static folder exists
# os.makedirs(os.path.join('static', 'uploaded_images'), exist_ok=True)

# # Path for emotion-based songs
# EMOTION_SONGS = {
#     'happy': 'static/emotion_songs/happy/happy_song.mp3',
#     'calm': 'static/emotion_songs/calm/calm_song.mp3',
#     'angry': 'static/emotion_songs/angry/angry_song.mp3',
#     'sad': 'static/emotion_songs/sad/sad_song.mp3'
# }

# # Function to detect emotion using DeepFace
# def detect_emotion(image_path):
#     try:
#         # Use DeepFace to analyze the emotion
#         result = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)
#         emotion = result[0]['dominant_emotion']
#         return emotion
#     except Exception as e:
#         print("Error in emotion detection:", e)
#         return None

# # Function to detect face and process the image with OpenCV
# def detect_face(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
    
#     # Convert the image to grayscale for face detection
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Load pre-trained Haar Cascade for face detection
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     # Draw rectangles around faces and annotate the emotion
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around face
#         # Crop face for emotion detection
#         face_crop = image[y:y+h, x:x+w]
#         cv2.imwrite("static/face_crop.jpg", face_crop)  # Save the cropped face image (optional)
#         # Get emotion for the cropped face
#         emotion = detect_emotion("static/face_crop.jpg")
#         # Add emotion label on the image
#         cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     # Save the result image with face rectangles and emotion labels
#     result_image_path = 'static/result_image.jpg'
#     cv2.imwrite(result_image_path, image)

#     return emotion, result_image_path

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file part'})

#     image_file = request.files['image']
    
#     if image_file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     # Save the uploaded image to the uploaded_images folder
#     image_path = os.path.join('static', 'uploaded_images', image_file.filename)
#     image_file.save(image_path)

#     # Detect faces and emotions in the uploaded image
#     emotion, result_image_path = detect_face(image_path)

#     # Get the corresponding song based on the emotion
#     if emotion in EMOTION_SONGS:
#         song_path = EMOTION_SONGS[emotion]
#     else:
#         song_path = None

#     return jsonify({'emotion': emotion, 'result_image': result_image_path, 'song': song_path})

# if __name__ == "__main__":
#     app.run(debug=True)