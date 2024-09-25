import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load an image from file
image_path = 'image.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image.")
    exit()

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Perform face detection
    results = face_detection.process(image_rgb)

    # Convert the RGB image back to BGR for displaying
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Draw detection results
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            # Get bounding box coordinates
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(image_bgr, bbox, (255, 0, 0), 6)
            # save image
            cv2.imwrite('detected_faces.jpg', image_bgr)
            print("Faces detected and saved as 'detected_faces.jpg'")

    # Display the resulting image
    cv2.imshow('Face Detection', image_bgr)
    cv2.waitKey(0)  # Wait for a key press to close the window

# Close windows
cv2.destroyAllWindows()
