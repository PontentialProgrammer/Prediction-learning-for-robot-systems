import cv2
import time
import numpy as np
import mujoco
import mujoco.viewer
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import the existing perception logic
from hand import extract_rl_state, MODEL_PATH

# Import the Shadow Hand model
from robot_descriptions import shadow_hand_mj_description

def map_human_to_robot(human_angles, num_actuators):
    """
    Hardcoded mapping of MediaPipe joint angles to Shadow Hand actuators.
    MediaPipe Right Hand angles are at indices 15-29.
    """
    ctrl = np.zeros(num_actuators)
    right_hand_angles = human_angles[15:30]
    
    # If hand is not detected, return zeros
    if np.all(right_hand_angles == 0):
        return ctrl

    # Helper function to convert MediaPipe angle (approx 180=straight, 90=bent) 
    # to MuJoCo actuator radian (approx 0.0=straight, 1.5=bent)
    def scale_angle(angle, max_rad=1.5):
        # Invert so 180 deg -> 0 rad, 90 deg -> max_rad
        # clip to ensure we don't break the physics limits
        normalized = np.clip((180.0 - angle) / 90.0, 0.0, 1.0)
        return normalized * max_rad

    # Shadow Hand Actuator Layout (from mujoco.mj_id2name):
    # Thumb: THJ5(2), THJ4(3), THJ3(4), THJ2(5), THJ1(6)
    # Index (FF): FFJ4(7), FFJ3(8), FFJ0(9)
    # Middle (MF): MFJ4(10), MFJ3(11), MFJ0(12)
    # Ring (RF): RFJ4(13), RFJ3(14), RFJ0(15)
    # Pinky (LF): LFJ5(16), LFJ4(17), LFJ3(18), LFJ0(19)

    # MediaPipe Right Hand Indices:
    # Thumb: 0 (Base), 1 (Middle), 2 (Tip)
    # Index: 3 (Knuckle), 4 (Middle), 5 (Base)
    # Middle: 6 (Knuckle), 7 (Middle), 8 (Base)
    # Ring: 9 (Knuckle), 10 (Middle), 11 (Base)
    # Pinky: 12 (Knuckle), 13 (Middle), 14 (Base)

    # --- HARDCODED MAPPING ---
    # Index Finger
    ctrl[8] = scale_angle(right_hand_angles[3])  # FFJ3 (Knuckle)
    ctrl[9] = scale_angle(right_hand_angles[4])  # FFJ0 (Middle/Tip)

    # Middle Finger
    ctrl[11] = scale_angle(right_hand_angles[6]) # MFJ3 (Knuckle)
    ctrl[12] = scale_angle(right_hand_angles[7]) # MFJ0 (Middle/Tip)

    # Ring Finger
    ctrl[14] = scale_angle(right_hand_angles[9]) # RFJ3 (Knuckle)
    ctrl[15] = scale_angle(right_hand_angles[10]) # RFJ0 (Middle/Tip)

    # Pinky Finger
    ctrl[18] = scale_angle(right_hand_angles[12]) # LFJ3 (Knuckle)
    ctrl[19] = scale_angle(right_hand_angles[13]) # LFJ0 (Middle/Tip)

    # Thumb (Mapping thumb is notoriously tricky due to axis rotation, 
    # but we map the primary flex bends here)
    ctrl[4] = scale_angle(right_hand_angles[1]) # THJ3 
    ctrl[3] = scale_angle(right_hand_angles[2]) # THJ4

    return ctrl

def main():
    # 1. Initialize MuJoCo Model
    model = mujoco.MjModel.from_xml_path(shadow_hand_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)
    
    # 2. Initialize OpenCV / MediaPipe
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

    # 3. Start the synchronous loop
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            
            # Close the viewer when you hit ESC in the OpenCV window, or when the viewer is closed
            while viewer.is_running() and cap.isOpened():
                step_start = time.time()
                
                success, frame = cap.read()
                if not success:
                    continue

                # --- Perception Step ---
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                
                # Async detection
                landmarker.detect_async(mp_image, timestamp)
                
                # Get the state vector (30-element array)
                rl_observation_space = extract_rl_state(latest_result)
                
                # --- Control/Mapping Step ---
                # Map the human joint angles directly to the MuJoCo actuator commands
                ctrl_command = map_human_to_robot(rl_observation_space, model.nu)
                data.ctrl[:] = ctrl_command
                
                # --- Simulation Step ---
                # Step the physics simulation
                mujoco.mj_step(model, data)
                
                # Update the 3D viewer
                viewer.sync()

                # Show the webcam feed
                cv2.imshow('RL Observation Stream', frame)
                if cv2.waitKey(1) & 0xFF == 27: # ESC to quit
                    break
                    
                # Timekeeping to match physics timestep (default ~2ms)
                # For real-time sync with camera (usually 30fps = ~33ms), we might need to step multiple times
                # But for a simple prototype, stepping once per camera frame is okay for testing the mapping.
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
