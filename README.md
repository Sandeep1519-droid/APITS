# Autonomous Projectile Interception and Tracking System (APITS) v6.0

An advanced, real-time spatial tracking and trajectory prediction architecture designed for edge hardware. APITS fuses neuromorphic event tracking with traditional texture-based computer vision to maintain target lock through high-speed motion blur, occlusion, and heavy ego-motion.

## 🚀 System Architecture & Key Features

* **9D Unscented Kalman Filter (UKF):** A custom mathematical state estimator (`ukf_core.py`) that isolates optical jitter and projects a stable 3D reality, tracking target position, velocity, and acceleration.
* **Asymmetric Spatio-Temporal Fusion (ASTF):** Dynamically hands off tracking duties between a texture-based CSRT tracker and a fallback neuromorphic Spiking Neural Network (SNN) based on real-time confidence metrics.
* **Physics-Anchored Neural Blend (PANB):** Overcomes standard sequence-to-sequence model exposure bias by computing a kinematic baseline and applying scaled, ONNX-accelerated neural residuals (`omega_brain.py`).
* **Ego-Motion Compensation:** Utilizes sparse LK optical flow and RANSAC homography to cancel out background movement when the camera turret is active.
* **Shadow Mode Telemetry:** Automatically records and exports rich time-series data and system ping metrics to CSV for offline model retraining.

## 🎬 Demonstration

You can view the system tracking targets through occlusions in the `demos/` directory. 
* [Watch: Drone Tracking & Occlusion Recovery (MP4)](./demos/Drone_Flies_Behind_Pillar.mp4)

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Computer Vision:** OpenCV (SimpleBlobDetector, CSRT, LK Optical Flow)
* **AI / Deep Learning:** PyTorch, ONNX Runtime, Custom LIF Neurons
* **Math & Data:** NumPy, Joblib

## ⚙️ Installation & Usage

1. **Clone the repository:**
   *(Note: This project uses Git LFS for the ONNX neural network weights. Ensure you have Git LFS installed before cloning).*
   ```bash
   git lfs install
   git clone [https://github.com/Sandeep1519-droid/APITS.git](https://github.com/Sandeep1519-droid/APITS.git)
   cd APITS
   
2. Install Dependencies:
   pip install opencv-python numpy torch onnxruntime joblib
   
3. Run the Master System:
   python main.py
