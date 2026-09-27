import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# Index mappings for calculating finger joint angles (Prev, Joint, Next)
ANGLE_DEFINITIONS = [
    # Thumb: Base, Middle, Tip
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    # Index: Knuckle, Middle, Lower Tip
    (5, 6, 7), (6, 7, 8), (0, 5, 6),
    # Middle
    (9, 10, 11), (10, 11, 12), (0, 9, 10),
    # Ring
    (13, 14, 15), (14, 15, 16), (0, 13, 14),
    # Pinky
    (17, 18, 19), (18, 19, 20), (0, 17, 18)
]

def calculate_3d_angle(p_prev, p_joint, p_next):
    a = np.array([p_prev.x, p_prev.y, p_prev.z])
    b = np.array([p_joint.x, p_joint.y, p_joint.z])
    c = np.array([p_next.x, p_next.y, p_next.z])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def extract_rl_state(latest_result):
    """
    Outputs a flat state vector of size 30: 
    [15 angles Left Hand, 15 angles Right Hand].
    Missing hands are padded with zeros to preserve shape.
    """
    state_vector = np.zeros(30, dtype=np.float32)
    
    if latest_result is None or not latest_result.hand_world_landmarks:
        return state_vector

    # Parse what hands were detected
    for idx, handedness in enumerate(latest_result.handedness):
        hand_label = handedness[0].category_name # "Left" or "Right"
        world_landmarks = latest_result.hand_world_landmarks[idx]
        
        # Calculate the 14 defined angles for this hand
        hand_angles = []
        for prev_i, joint_i, next_i in ANGLE_DEFINITIONS:
            angle = calculate_3d_angle(
                world_landmarks[prev_i], 
                world_landmarks[joint_i], 
                world_landmarks[next_i]
            )
            hand_angles.append(angle)
            
        # Place hand data in designated slots inside the vector
        if hand_label == "Left":
            state_vector[0:15] = hand_angles
        elif hand_label == "Right":
            state_vector[15:30] = hand_angles
            
    return state_vector

def main():
    # --- Live Feed Loop ---
    cap = cv2.VideoCapture(0)
    latest_result = None

    def async_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        nonlocal latest_result
        latest_result = result

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=async_callback
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: continue

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            landmarker.detect_async(mp_image, timestamp)

            # Construct your state vector
            rl_observation_space = extract_rl_state(latest_result)
            
            # --- SEND TO YOUR AGENT HERE ---
            # Example: agent.step(rl_observation_space)
            # For testing, we print the array shape and a subset of the angles
            print(f"RL State Vector Shape: {rl_observation_space.shape} | Sample Angle: {rl_observation_space[0]:.1f}°")

            cv2.imshow('RL Observation Stream', frame)
            if cv2.waitKey(1) & 0xFF == 27: 
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
