import tensorflow as tf
import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_recognize_faces(image_path, known_face_paths):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load TensorFlow model
    model = tf.saved_model.load('https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1')
    detector = model.signatures['default']
    
    # Prepare the image for detection
    image_tensor = tf.convert_to_tensor(image_rgb)
    image_tensor = tf.image.convert_image_dtype(image_tensor, dtype=tf.uint8)
    image_tensor = tf.expand_dims(image_tensor, 0)
    
    # Perform detection
    results = detector(image_tensor)
    boxes = results['detection_boxes'].numpy()[0]
    scores = results['detection_scores'].numpy()[0]
    
    # Load known faces
    known_face_encodings = []
    known_face_names = []
    
    for path in known_face_paths:
        known_image = face_recognition.load_image_file(path)
        known_face_encoding = face_recognition.face_encodings(known_image)[0]
        known_face_encodings.append(known_face_encoding)
        known_face_names.append(path.split('/')[-1].split('.')[0])  # Use filename as the name
    
    # Initialize face locations
    height, width, _ = image.shape
    for box, score in zip(boxes, scores):
        if score > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            face_image = image[ymin:ymax, xmin:xmax]
            face_encoding = face_recognition.face_encodings(face_image)
            
            if face_encoding:
                face_encoding = face_encoding[0]
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                
                # Draw rectangle and label
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(image, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the result
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage
detect_and_recognize_faces('path_to_image.jpg', ['known_face1.jpg', 'known_face2.jpg'])

