# Predictive Robotic Hand Control

This repository contains the foundation for a real-time, predictive robotic hand control system. The project bridges computer vision (MediaPipe) with a rigid-body physics simulation (MuJoCo), serving as the first step towards integrating a General Value Function (GVF) Reinforcement Learning layer.

## Architecture

1. **Perception Layer (`hand.py`):** 
   - Uses OpenCV and MediaPipe's `HandLandmarker` task to extract 3D hand landmarks from a live webcam feed.
   - Calculates 15 specific joint angles (in degrees) for the human hand in real-time.

2. **Simulation & Mapping Layer (`sim.py`):**
   - Integrates the DeepMind Shadow Dexterous Hand model into a MuJoCo passive viewer.
   - Applies a deterministic mathematical mapping (`map_human_to_robot`) to translate the human hand angles into the 20+ specific actuator radian limits of the simulated Shadow Hand.
   - Synchronizes the computer vision pipeline with the physics engine to actuate the robotic hand in real-time.

## Next Steps: The RL / GVF Layer
The current direct mathematical mapping proves the end-to-end data pipeline is functional, but manual mapping across mismatched human-robot morphologies is inherently noisy. 

The next phase introduces a Reinforcement Learning agent (Proximal Policy Optimization) between the perception output and the actuator input. This agent will utilize General Value Functions (GVFs) to *anticipate* user intent based on the continuous sensory stream, allowing the robotic hand to seamlessly co-adapt and pre-actuate grips.

## Requirements
- `opencv-python`
- `mediapipe`
- `mujoco`
- `robot_descriptions`

## Usage
Run the integrated simulation environment:
```bash
python sim.py
```
*(Ensure `hand_landmarker.task` is present in the root directory)*
