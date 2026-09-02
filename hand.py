import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
MODEL_PATH = "hand_landmarker.task"  # Make sure this file is in your folder!

# Helper to draw landmarks (The Tasks API doesn't bundle drawing styles anymore, 
# so we draw lines and points using basic OpenCV functions)
# A complete hardcoded list mapping hand joint connections (replacing mp_hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Knuckle connections (Palm baseline)
    (5, 9), (9, 13), (13, 17)
]

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = rgb_image.copy()
    h, w, _ = rgb_image.shape

    for hand_landmarks in hand_landmarks_list:
        # Convert landmarks from normalized coordinates to actual pixel coordinates
        pixel_points = []
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            pixel_points.append((cx, cy))

        # 1. First draw the skeleton lines connecting the joints
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            cv2.line(annotated_image, pixel_points[start_idx], pixel_points[end_idx], (255, 0, 0), 2) # Blue lines

        # 2. Then draw the joint marker circles on top
        for point in pixel_points:
            cv2.circle(annotated_image, point, 5, (0, 255, 0), -1) # Green markers
            
    return annotated_image


# ==========================================
# PART 1: STATIC IMAGES
# ==========================================
IMAGE_FILES = []  # Add your image paths here if needed

if IMAGE_FILES:
    # Configure options for static images
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.flip(cv2.imread(file), 1)
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # MediaPipe Tasks requires its own Image wrapper object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Process the image
            detection_result = landmarker.detect(mp_image)
            
            # Draw and save results
            if detection_result.hand_landmarks:
                print('Handedness:', detection_result.handedness)
                annotated_image = draw_landmarks_on_image(rgb_image, detection_result)
                # Convert back to BGR for saving
                bgr_annotated = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'/tmp/annotated_image{idx}.png', cv2.flip(bgr_annotated, 1))

# ==========================================
# PART 2: WEBCAM INPUT (LIVE STREAM MODE)
# ==========================================
cap = cv2.VideoCapture(0)

# Configure options for Live Streaming. 
# Live stream mode requires a callback function to handle async results.
latest_result = None

def save_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=save_result
)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        # Get system timestamp in milliseconds (Required for live stream mode)
        frame_timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        # Send frame to the landmarker asynchronously
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Draw the latest available results
        display_image = image
        if latest_result is not None and latest_result.hand_landmarks:
            annotated_rgb = draw_landmarks_on_image(rgb_image, latest_result)
            display_image = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Flip horizontally for selfie-view
        cv2.imshow('MediaPipe Tasks Hands', cv2.flip(display_image, 1))
        
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
            break

cap.release()
cv2.destroyAllWindows()
