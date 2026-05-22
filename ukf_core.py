import numpy as np
import math

# ============================================================
# TIER-1 DEFENSE UPGRADE: 9D UNSCENTED KALMAN FILTER (UKF)
# ============================================================
# This engine isolates the optical jitter from YOLO and 
# mathematically hallucinates a perfect 3D reality.
# State Vector: [X, Y, Z, Vx, Vy, Vz, Ax, Ay, Az]
# Measurement Vector: [cam_x, cam_y, cam_w]
# ============================================================

class MonocularUKF:
    def __init__(self, dt=0.033, focal_length=600.0, real_width=0.067):
        self.dt = dt
        self.f = focal_length
        self.real_w = real_width

        # 9D State Vector: [X, Y, Z, Vx, Vy, Vz, Ax, Ay, Az]
        self.x = np.zeros(9)
        self.x[2] = 5.0 # Initial guess: Target is 5 meters away

        # Covariance Matrix (P) - The "Uncertainty" of our state
        self.P = np.eye(9) * 500.0

        # Process Noise (Q) - How chaotic the drone's actual flight is
        self.Q = np.eye(9) * 0.1
        self.Q[6:9, 6:9] *= 10.0 # High variance for acceleration (drones zip around)

        # Measurement Noise (R) - How much we trust the YOLO bounding box
        # We assume YOLO has ~5px jitter on X/Y, and ~10px jitter on Width
        self.R = np.diag([5.0, 5.0, 10.0]) 

        # Unscented Transform Parameters (Julier-Uhlmann method)
        self.alpha = 1e-3
        self.kappa = 0
        self.beta = 2
        self.n = 9
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n

        # Generate Weights for the 19 Sigma Points
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lam)))
        self.Wc = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lam)))
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

    def generate_sigma_points(self):
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = self.x
        
        # Calculate the square root of the covariance matrix
        # 1. Enforce mathematical symmetry (fixes floating point drift)
        self.P = (self.P + self.P.T) / 2.0
        
        # 2. Add microscopic Covariance Jitter to guarantee positive-definite state
        self.P = self.P + np.eye(self.n) * 1e-6
        
        # 3. Safe Cholesky Decomposition
        try:
            U = np.linalg.cholesky((self.n + self.lam) * self.P)
        except np.linalg.LinAlgError:
            # Absolute worst-case fallback: Reset the covariance matrix if it completely shatters
            self.P = np.eye(self.n) * 1.0
            U = np.linalg.cholesky((self.n + self.lam) * self.P)
        
        for i in range(self.n):
            sigmas[i + 1] = self.x + U[i]
            sigmas[self.n + i + 1] = self.x - U[i]
        return sigmas

    def state_transition(self, sigmas):
        # Apply strict 3D Newtonian Physics to all 19 Sigma Points
        new_sigmas = np.zeros_like(sigmas)
        dt = self.dt
        for i, s in enumerate(sigmas):
            # Position = Position + Velocity*dt + 0.5*Acceleration*dt^2
            new_sigmas[i, 0:3] = s[0:3] + s[3:6]*dt + 0.5*s[6:9]*(dt**2)
            # Velocity = Velocity + Acceleration*dt
            new_sigmas[i, 3:6] = s[3:6] + s[6:9]*dt
            # Acceleration remains constant for this micro-step
            new_sigmas[i, 6:9] = s[6:9]
        return new_sigmas

    def measurement_model(self, sigmas, cx, cy):
        # Project the 3D Sigma Points back through the 2D camera lens
        Z_sigmas = np.zeros((sigmas.shape[0], 3))
        for i, s in enumerate(sigmas):
            X, Y, Z = s[0], s[1], s[2]
            if Z < 0.1: Z = 0.1 # Prevent mathematical singularity
            
            px_x = (self.f * X / Z) + cx
            px_y = (self.f * Y / Z) + cy
            px_w = (self.f * self.real_w) / Z
            
            Z_sigmas[i] = [px_x, px_y, px_w]
        return Z_sigmas

    def predict(self):
        # 1. Generate the quantum cloud of points
        sigmas = self.generate_sigma_points()
        
        # 2. Push them through physical time
        self.sigmas_f = self.state_transition(sigmas)

        # 3. Collapse the cloud back into a single predicted state
        self.x = np.dot(self.Wm, self.sigmas_f)

        self.P = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            y = self.sigmas_f[i] - self.x
            self.P += self.Wc[i] * np.outer(y, y)
        self.P += self.Q

    def update(self, measurement, frame_w, frame_h):
        cx, cy = frame_w / 2.0, frame_h / 2.0

        # Transform the 3D points into 2D camera pixel guesses
        Z_sigmas = self.measurement_model(self.sigmas_f, cx, cy)

        # Calculate the single most likely 2D pixel coordinate
        zp = np.dot(self.Wm, Z_sigmas)

        # Covariance of measurement
        Pz = np.zeros((3, 3))
        for i in range(2 * self.n + 1):
            y = Z_sigmas[i] - zp
            Pz += self.Wc[i] * np.outer(y, y)
        Pz += self.R

        # Cross covariance (Bridging 2D pixels with 3D space)
        Pxz = np.zeros((self.n, 3))
        for i in range(2 * self.n + 1):
            y_state = self.sigmas_f[i] - self.x
            y_meas = Z_sigmas[i] - zp
            Pxz += self.Wc[i] * np.outer(y_state, y_meas)

        # The Kalman Gain (The Ultimate Truth Matrix)
        K = np.dot(Pxz, np.linalg.inv(Pz))

        # Update the final 9D State based on the YOLO reality
        y = measurement - zp
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(Pz, K.T))

        # Return: [X, Y, Z, Vx, Vy, Vz, Ax, Ay, Az]
        return self.x