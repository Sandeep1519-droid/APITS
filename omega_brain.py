import numpy as np
import joblib
import time
import math
from collections import deque
import os
import onnxruntime as ort

# ============================================================
# PHYSICS-ANCHORED NEURAL TRAJECTORY ENGINE
# ============================================================
# Root cause of the ONNX failure:
#
# The Seq2Seq model was trained on MOVING ball data. When the
# ball is slow or nearly stationary, the normalized input
# features fall completely outside the training distribution.
# The model panics and outputs large offsets (e.g. -4.3 * 100
# = -430 pixels on step 1), sending the trajectory off-screen.
#
# This is the "exposure bias + OOD" double failure:
#   - Exposure bias:  decoder feeds its own wrong predictions
#                     back, cascading the error across 10 steps
#   - Out-of-distribution: near-zero velocities were never in
#                     the training data so the model guesses
#
# Solution: Physics-Anchored Neural Blend (PANB)
#   1. Compute a kinematic parabola (always correct)
#   2. Run ONNX and get neural residuals
#   3. Apply Epsilon Scaling: scale neural magnitude DOWN by
#      the current speed ratio (slow ball → tiny neural weight)
#   4. Clip each neural offset to the kinematic neighborhood
#      so it can never fly off-screen regardless of model error
#   5. EMA smooth the final result to kill any residual flicker
# ============================================================

class OmegaKinematicEngine:
    def __init__(self, model_path='omega_engine.onnx', scaler_path='omega_scaler.pkl'):
        print("[BRAIN] Engaging OMEGA PANB Core (Physics-Anchored Neural Blend)...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"CRITICAL: '{model_path}' not found.\n"
                "  -> Run compile_onnx.py first."
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"CRITICAL: '{scaler_path}' not found.\n"
                "  -> Run train_omega.py first."
            )

        self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_names = [inp.name for inp in self.ort_session.get_inputs()]
        self.scaler      = joblib.load(scaler_path)

        # Sensor history
        self.buffer   = deque(maxlen=20)
        self.z_buffer = deque(maxlen=10)
        self.last_time = time.perf_counter()

        # Smoothing state — in PIXEL COORDINATE space (after reconstruction)
        self.smoothed_path = None    # list of (fx, fy) — 15 points
        self.EMA_ALPHA     = 0.30    # 30% new, 70% previous — tune if needed

        # Physics reference speed (px/s).
        # Neural contribution scales from 0 at rest to MAX_NEURAL_WEIGHT
        # when the ball exceeds this speed. Calibrate to your scene.
        self.REFERENCE_SPEED_PX  = 150.0   # px/s at which neural weight = max
        self.MAX_NEURAL_WEIGHT   = 0.20    # never let neural dominate

        # 3D geometry
        self.FOCAL_LENGTH         = 600.0
        self.REAL_WIDTH           = 0.067
        self.INTERCEPTOR_SPEED_MS = 150.0

        self.last_valid_path = []
        print("[BRAIN] PANB core ready.")

    # -------------------------------------------------------
    def update(self, cam_x, cam_y, cam_w, frame_w, frame_h, neuron_spiked):
        now = time.perf_counter()
        dt  = max(now - self.last_time, 0.001)
        self.last_time = now

        if cam_x == 0 and cam_y == 0:
            self.buffer.clear()
            self.z_buffer.clear()
            self.smoothed_path = None
            return [], (0, 0, 0), (0, 0, 0), 0.0, 0.0, False, "SEARCHING"

        self.buffer.append([cam_x, cam_y, dt])

        current_z = (self.REAL_WIDTH * self.FOCAL_LENGTH) / max(cam_w, 1.0)
        self.z_buffer.append([current_z, dt])

        if len(self.buffer) < 15:
            return [], (0, 0, 0), (0, 0, 0), 0.0, 0.0, False, "OMEGA BOOTING"

        try:
            return self._run_inference(cam_x, cam_y, current_z, frame_w, frame_h)
        except Exception as e:
            print(f"\n[!!!] PANB SAFEGUARD: {e}\n")
            return self.last_valid_path, (0, 0, 0), (0, 0, 0), 0.0, 0.0, False, "OMEGA RECOVERING"

    # -------------------------------------------------------
    def _run_inference(self, cam_x, cam_y, current_z, frame_w, frame_h):
        hist = np.array(self.buffer)   # (N, 3): [x, y, dt]

        # --- 1. Kinematic features ---
        vx = np.diff(hist[:, 0]) / hist[1:, 2]
        vy = np.diff(hist[:, 1]) / hist[1:, 2]
        ax = np.diff(vx) / hist[2:, 2]
        ay = np.diff(vy) / hist[2:, 2]
        jx = np.diff(ax) / hist[3:, 2]
        jy = np.diff(ay) / hist[3:, 2]

        seq    = np.column_stack((vx[-10:], vy[-10:], ax[-10:], ay[-10:], jx[-10:], jy[-10:]))
        dt_seq = hist[-10:, 2].reshape(-1, 1).astype(np.float32)

        seq    = np.nan_to_num(seq,    nan=0.0, posinf=0.0, neginf=0.0)
        dt_seq = np.nan_to_num(dt_seq, nan=0.033, posinf=0.033, neginf=0.033)

        scaled_seq = self.scaler.transform(seq).astype(np.float32)
        scaled_seq = np.nan_to_num(scaled_seq, nan=0.0, posinf=0.0, neginf=0.0)

        avg_dt = float(np.mean(dt_seq))
        avg_vx = float(np.mean(vx[-3:]))
        avg_vy = float(np.mean(vy[-3:]))
        avg_ax = float(np.mean(ax[-3:]))
        avg_ay = float(np.mean(ay[-3:]))

        # =================================================================
        # 2. PHYSICS BACKBONE — always correct, no distribution issues
        # =================================================================
        # 15-point kinematic parabola anchored to current ball position.
        # This alone would give a solid working trajectory. The neural
        # network will only add a small bounded correction on top.
        kin_offsets = []   # (dx, dy) in pixels relative to cam_x, cam_y
        for step in range(1, 16):
            t   = step * avg_dt
            kdx = avg_vx * t + 0.5 * avg_ax * (t ** 2)
            kdy = avg_vy * t + 0.5 * avg_ay * (t ** 2)
            kin_offsets.append((kdx, kdy))

        # =================================================================
        # 3. ONNX NEURAL OUTPUT
        # =================================================================
        future_dt = np.full((1, 10, 1), avg_dt, dtype=np.float32)   # x1.0 — trained range

        ort_inputs = {
            self.input_names[0]: np.expand_dims(scaled_seq, axis=0),
            self.input_names[1]: np.expand_dims(dt_seq, axis=0),
            self.input_names[2]: future_dt,
        }
        ort_out       = self.ort_session.run(None, ort_inputs)
        raw_offsets   = np.array(ort_out[0][0], dtype=float)   # (10, 2) — divided by 100 in training

        # =================================================================
        # 4. EPSILON SCALING  (training-free exposure-bias correction)
        # =================================================================
        # Scale the neural output DOWN proportionally to how far the
        # current speed is below the reference speed. This keeps the
        # neural correction within the magnitude range seen during training.
        # When the ball is fast (in-distribution), neural weight rises.
        # When the ball is slow/stationary (OOD), neural weight → 0.
        current_speed_px = math.sqrt(avg_vx ** 2 + avg_vy ** 2)
        epsilon = np.clip(current_speed_px / self.REFERENCE_SPEED_PX, 0.0, 1.0)
        neural_weight = epsilon * self.MAX_NEURAL_WEIGHT   # 0 → 0.20 range

        # =================================================================
        # 5. PHYSICS-ANCHORED BLEND with hard clipping
        # =================================================================
        # For each of the first 10 steps: blend kinematic offset (dominant)
        # with the neural residual (small, bounded). For steps 11-15 we
        # extrapolate beyond the neural horizon using kinematics only.
        raw_path = []

        for i in range(15):
            kin_dx, kin_dy = kin_offsets[i]

            if i < 10:
                # Neural offset (×100 to undo the training normalisation)
                n_dx = raw_offsets[i][0] * 100.0
                n_dy = raw_offsets[i][1] * 100.0

                # Hard clip: neural offset can't exceed ±2× kinematic
                # displacement at this step OR a minimum of 30px.
                # This prevents off-screen lunges regardless of model error.
                clip_x = max(abs(kin_dx) * 2.0, 30.0)
                clip_y = max(abs(kin_dy) * 2.0, 30.0)
                n_dx   = np.clip(n_dx, -clip_x, clip_x)
                n_dy   = np.clip(n_dy, -clip_y, clip_y)

                # Final blend: physics backbone + bounded neural residual
                final_dx = (1.0 - neural_weight) * kin_dx + neural_weight * n_dx
                final_dy = (1.0 - neural_weight) * kin_dy + neural_weight * n_dy
            else:
                # Beyond neural horizon: pure kinematics
                final_dx = kin_dx
                final_dy = kin_dy

            raw_path.append((int(cam_x + final_dx), int(cam_y + final_dy)))

        # =================================================================
        # 6. EMA SMOOTHING across frames — eliminates residual flicker
        # =================================================================
        if self.smoothed_path is None or len(self.smoothed_path) != 15:
            self.smoothed_path = raw_path[:]
        else:
            smoothed = []
            for (rx, ry), (sx, sy) in zip(raw_path, self.smoothed_path):
                sx_new = int(self.EMA_ALPHA * rx + (1.0 - self.EMA_ALPHA) * sx)
                sy_new = int(self.EMA_ALPHA * ry + (1.0 - self.EMA_ALPHA) * sy)
                smoothed.append((sx_new, sy_new))
            self.smoothed_path = smoothed

        future_path = self.smoothed_path

        # =================================================================
        # 7. 3D KINEMATICS & KILL CHAIN (unchanged)
        # =================================================================
        z_hist  = np.array(self.z_buffer)
        avg_vz  = (np.mean(np.diff(z_hist[:, 0]) / z_hist[1:, 1])
                   if len(z_hist) > 2 else 0.0)

        lead_time = avg_dt * 15.0
        future_z  = current_z + avg_vz * lead_time

        cx, cy = frame_w / 2.0, frame_h / 2.0

        real_x = ((cam_x - cx) * current_z) / self.FOCAL_LENGTH
        real_y = ((cam_y - cy) * current_z) / self.FOCAL_LENGTH

        end_px, end_py = future_path[-1]
        fut_real_x = ((end_px - cx) * future_z) / self.FOCAL_LENGTH
        fut_real_y = ((end_py - cy) * future_z) / self.FOCAL_LENGTH

        physical_speed_px = current_speed_px
        speed_3d = math.sqrt(
            ((avg_vx * current_z) / self.FOCAL_LENGTH) ** 2 +
            ((avg_vy * current_z) / self.FOCAL_LENGTH) ** 2 +
            avg_vz ** 2
        )

        is_stationary = physical_speed_px < 15.0
        status_flag   = "HOVER LOCKED" if is_stationary else "OMEGA LOCKED"

        dist_to_impact = math.sqrt(fut_real_x ** 2 + fut_real_y ** 2 + future_z ** 2)
        tti            = dist_to_impact / max(self.INTERCEPTOR_SPEED_MS, 1.0)
        launch_auth    = (tti <= lead_time or is_stationary) and current_z > 0

        self.last_valid_path = future_path
        return (
            future_path,
            (real_x,     real_y,     current_z),
            (fut_real_x, fut_real_y, future_z),
            speed_3d,
            tti,
            launch_auth,
            status_flag,
        )
