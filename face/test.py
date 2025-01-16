import cv2
from ultralytics import YOLO
import mediapipe as mp

# Load YOLO model
model = YOLO("best.pt")  # Load a custom trained YOLO model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Open the video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands using MediaPipe
    results_hands = hands.process(frame_rgb)

    # Annotated frame for visualization
    annotated_frame = frame.copy()

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            # Get the bounding box of the hand
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Ensure bounding box is within frame boundaries
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            # Crop the hand region from the frame
            hand_region = frame[y_min:y_max, x_min:x_max]

            # Skip empty or invalid regions
            if hand_region.size == 0:
                continue

            # Convert the cropped hand region to RGB for YOLO
            hand_region_rgb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB)

            # Perform YOLO inference on the cropped hand region
            try:
                results_yolo = model(hand_region_rgb)
            except Exception as e:
                print(f"YOLO error: {e}")
                continue

            # Visualize YOLO detection results
            if results_yolo and results_yolo[0].boxes:
                for det in results_yolo[0].boxes:
                    label = model.names[int(det.cls)]  # Class label
                    conf = det.conf  # Confidence
                    cv2.putText(
                        annotated_frame,
                        f"{label} ({conf:.2f})",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            # Draw MediaPipe hand landmarks on the annotated frame
            mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the annotated frame
    cv2.imshow('Hand Gesture Recognition', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
