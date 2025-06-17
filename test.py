import cv2
import mediapipe as mp

# Load MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 68 landmark indices from the 468 MediaPipe points (approximation of dlib's 68)
LANDMARK_IDS_68 = [
    # Jawline
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
    152, 377, 400, 378, 379, 365, 397,

    # Right eyebrow
    70, 63, 105, 66, 107,

    # Left eyebrow
    336, 296, 334, 293, 300,

    # Nose bridge
    168, 6, 197, 195,

    # Lower nose
    5, 4, 1, 19, 94,

    # Right eye
    33, 160, 158, 133, 153, 144,

    # Left eye
    362, 385, 387, 263, 373, 380,

    # Outer lips
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,

    # Inner lips
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]

def extract_and_display_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image.")
        return

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            print("No face detected.")
            return

        h, w, _ = image.shape
        for face_landmarks in results.multi_face_landmarks:
            landmark_coords = []
            for idx in LANDMARK_IDS_68:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                landmark_coords.append((x, y))
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

            # Print landmarks as string
            landmark_string = ', '.join([f'({x},{y})' for x, y in landmark_coords])
            print("68 Landmarks:\n", landmark_string)

        # Show and save output image
        cv2.imshow("Face Landmarks", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite("output_with_landmarks.jpg", image)

# Example usage
extract_and_display_landmarks("input.jpeg")
