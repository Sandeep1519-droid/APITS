import numpy as np
import torch
import torch.nn as nn
from collections import deque
import os
import joblib
import math

# --- 1. THE SNN TRIGGER ---
class AdaptiveLIFNeuron:
    def __init__(self, leak_rate=0.85, base_threshold=100.0):
        self.v_mem = 0.0              
        self.leak_rate = leak_rate    
        self.base_threshold = base_threshold
        self.current_threshold = base_threshold

    def process_stimulus(self, motion_area):
        if motion_area <= 0:
            self.v_mem *= (self.leak_rate * 0.5)
            return False

        input_current = np.sqrt(motion_area) * 2.0 
        self.v_mem = (self.v_mem * self.leak_rate) + input_current
        self.current_threshold = self.base_threshold + (input_current * 0.1)

        if self.v_mem >= self.current_threshold:
            self.v_mem = 0.0 
            return True
            
        return False

# --- 2. THE APEX AI ---
class ApexMHA_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(ApexMHA_GRU, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, output_dim))

    def forward(self, x):
        x = self.input_projection(x)
        gru_out, _ = self.gru(x)
        attn_out, _ = self.mha(gru_out, gru_out, gru_out)
        return self.fc(attn_out[:, -1, :])

# --- 3. THE HYBRID FUSION ENGINE ---
class CustomKinematicEngine:
    def __init__(self):
        self.is_locked = False
        self.coast_frames = 0
        self.MAX_COAST_FRAMES = 60
        self.ai_enabled = False
        
        self.smooth_x = None
        self.smooth_y = None
        
        if os.path.exists("vajra_apex_weights.pth") and os.path.exists("vajra_scaler.pkl"):
            try:
                self.model = ApexMHA_GRU(9, 128, 3, 4)
                self.model.load_state_dict(torch.load("vajra_apex_weights.pth", weights_only=True))
                self.model.eval()
                self.scaler = joblib.load("vajra_scaler.pkl")
                self.ai_enabled = True
                self.buffer = deque(maxlen=13)
                self.last_pred_delta = np.zeros(3)
                print("[BRAIN] APEX LSTM Loaded! Engaging Neural Trajectory Curves.")
            except Exception as e:
                print(f"[BRAIN] Failed to load AI: {e}. Falling back to Kalman Filter.")
        else:
            print("[BRAIN] AI weights not found. Falling back to Linear Kalman Filter.")
            
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)

    def initialize_lock(self, target_x, target_y):
        self.smooth_x = float(target_x)
        self.smooth_y = float(target_y)
        self.x = np.array([[target_x], [target_y], [0.0], [0.0]], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64)
        self.is_locked = True
        self.coast_frames = 0
        if self.ai_enabled: self.buffer.clear()

    def update(self, cam_x, cam_y, neuron_spiked):
        if not self.is_locked:
            if neuron_spiked: self.initialize_lock(cam_x, cam_y)
            return int(cam_x), int(cam_y), "SCANNING"

        if self.ai_enabled:
            # --- THE MOMENTUM BRIDGE (Fixes the Fan/Spiderweb) ---
            if not neuron_spiked:
                self.coast_frames += 1
                if self.coast_frames > self.MAX_COAST_FRAMES:
                    self.is_locked = False
                    return 0, 0, "LOST"
                
                if len(self.buffer) > 0:
                    # Physically push the internal memory forward along the curve!
                    self.smooth_x += self.last_pred_delta[0]
                    self.smooth_y += self.last_pred_delta[1]
                    self.buffer.append(np.array([self.smooth_x, self.smooth_y, 0.0]))
                    
                    fx = int(self.smooth_x + (self.last_pred_delta[0] * 3))
                    fy = int(self.smooth_y + (self.last_pred_delta[1] * 3))
                    return fx, fy, "COASTING"
                return 0, 0, "LOST"
                
            self.coast_frames = 0
            
            # Anti-Teleportation Clamp
            if self.smooth_x is None:
                self.smooth_x, self.smooth_y = float(cam_x), float(cam_y)
            else:
                dist = math.hypot(cam_x - self.smooth_x, cam_y - self.smooth_y)
                alpha = 0.05 if dist > 100 else 0.25
                self.smooth_x = (alpha * cam_x) + ((1 - alpha) * self.smooth_x)
                self.smooth_y = (alpha * cam_y) + ((1 - alpha) * self.smooth_y)
            
            self.buffer.append(np.array([self.smooth_x, self.smooth_y, 0.0]))
            
            if len(self.buffer) < 13:
                return int(cam_x), int(cam_y), "LOCKED (BUFFERING)"
                
            v = np.diff(np.array(self.buffer), axis=0)
            a = np.diff(v, axis=0)
            j = np.diff(a, axis=0)
            kinematics = np.hstack((v[2:], a[1:], j))
            
            try:
                scaled_kinematics = self.scaler.transform(kinematics)
                x_tensor = torch.tensor(scaled_kinematics, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    pred_scaled = self.model(x_tensor).numpy()
                    
                dummy_array = np.zeros((1, 9))
                dummy_array[:, :3] = pred_scaled
                self.last_pred_delta = self.scaler.inverse_transform(dummy_array)[0, :3]
                
                future_x = int(self.smooth_x + (self.last_pred_delta[0] * 3)) 
                future_y = int(self.smooth_y + (self.last_pred_delta[1] * 3))
                return future_x, future_y, "LOCKED"
                
            except Exception as e:
                return int(cam_x), int(cam_y), "LOCKED (MATH ERROR)"

        # Fallback Linear Path
        self.x = np.dot(self.F, self.x)
        pred_x, pred_y = int(self.x[0, 0]), int(self.x[1, 0])

        if neuron_spiked:
            Z = np.array([[cam_x], [cam_y]], dtype=np.float64)
            y = Z - np.dot(self.H, self.x)
            K = np.dot(self.P, self.H.T) @ np.linalg.inv(np.dot(self.H, self.P) @ self.H.T + np.eye(2)*0.1)
            self.x = self.x + np.dot(K, y)
            self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)
            self.coast_frames = 0 
            return pred_x, pred_y, "LOCKED"
        else:
            self.coast_frames += 1
            if self.coast_frames > self.MAX_COAST_FRAMES:
                self.is_locked = False
                return 0, 0, "LOST"
            return pred_x, pred_y, "COASTING"