import cv2
import numpy as np
import time
import csv
import os
from collections import deque

class TacticalHUD:
    def __init__(self):
        self.session_start = time.perf_counter()
        self.last_frame_time = self.session_start
        self.current_dt = 0.0
        self.omega_dataset = deque(maxlen=100000) 
        self.trail = deque(maxlen=50) 
        
        self.C_CYAN = (255, 200, 0)     
        self.C_AMBER = (0, 255, 0) 
        self.C_RED = (0, 0, 255)
        self.C_DARK = (15, 15, 15)      
        self.C_GRAY = (100, 100, 100) 

        # --- SHADOW MODE VAULT ---
        self.log_folder = "telemetry_logs"
        os.makedirs(self.log_folder, exist_ok=True)

    def log_spacetime_telemetry(self, x, y, w, h, status):
        now = time.perf_counter()
        self.current_dt = now - self.last_frame_time
        self.last_frame_time = now
        mission_time = now - self.session_start
        if "LOCKED" in status and w > 0:
            self.omega_dataset.append({
                'time_sec': round(mission_time, 6), 'dt_sec': round(self.current_dt, 6),
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)
            })
        return self.current_dt

    def export_omega_dataset(self):
        if len(self.omega_dataset) < 100: return
        filename = os.path.join(self.log_folder, f"omega_training_data_{int(time.time())}.csv")
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['time_sec', 'dt_sec', 'x', 'y', 'w', 'h'])
            writer.writeheader()
            writer.writerows(self.omega_dataset)
        print(f"[SHADOW MODE] Flight data archived to: {filename}")

    def render(self, frame, target_x, target_y, target_w, target_h, future_path, curr_3d, fut_3d, speed_3d, tti, launch_auth, status):
        h_frame, w_frame = frame.shape[:2]
        
        if "LOCKED" in status:
            self.trail.append((int(target_x), int(target_y)))
            
            # --- PAST TRAIL ---
            if len(self.trail) > 1:
                for i in range(1, len(self.trail)):
                    alpha = i / len(self.trail) 
                    color = (int(self.C_CYAN[0]*alpha), int(self.C_CYAN[1]*alpha), int(self.C_CYAN[2]*alpha))
                    cv2.line(frame, self.trail[i-1], self.trail[i], color, 2)

            # --- TACTICAL BRACKETS ---
            rx, ry = int(target_x - target_w/2), int(target_y - target_h/2)
            corner_len = 15
            cv2.line(frame, (rx, ry), (rx + corner_len, ry), self.C_CYAN, 2)
            cv2.line(frame, (rx, ry), (rx, ry + corner_len), self.C_CYAN, 2)
            cv2.line(frame, (rx + target_w, ry), (rx + target_w - corner_len, ry), self.C_CYAN, 2)
            cv2.line(frame, (rx + target_w, ry), (rx + target_w, ry + corner_len), self.C_CYAN, 2)
            cv2.line(frame, (rx, ry + target_h), (rx + corner_len, ry + target_h), self.C_CYAN, 2)
            cv2.line(frame, (rx, ry + target_h), (rx, ry + target_h - corner_len), self.C_CYAN, 2)
            cv2.line(frame, (rx + target_w, ry + target_h), (rx + target_w - corner_len, ry + target_h), self.C_CYAN, 2)
            cv2.line(frame, (rx + target_w, ry + target_h), (rx + target_w, ry + target_h - corner_len), self.C_CYAN, 2)
            
            # ==========================================
            # TRUE-POINT AI TRAJECTORY RENDERER
            # ==========================================
            if future_path and len(future_path) > 1:
                try:
                    # [NEW FIX]: Only draw the sweeping line if it is ACTUALLY moving
                    if "HOVER" not in status:
                        for i in range(len(future_path) - 1):
                            pt1 = (int(future_path[i][0]), int(future_path[i][1]))
                            pt2 = (int(future_path[i+1][0]), int(future_path[i+1][1]))
                            
                            cv2.line(frame, pt1, pt2, (0, 70, 0), 6, cv2.LINE_AA)
                            cv2.line(frame, pt1, pt2, self.C_AMBER, 2, cv2.LINE_AA)
                        
                        for pt in future_path:
                            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (100, 255, 100), -1, cv2.LINE_AA)
                    
                    # Always draw the Impact Crosshair (If hovering, it drops right on the ball)
                    end_x, end_y = int(future_path[-1][0]), int(future_path[-1][1])
                    if launch_auth:
                        cv2.drawMarker(frame, (end_x, end_y), self.C_RED, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
                    else:
                        cv2.drawMarker(frame, (end_x, end_y), (0, 200, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)
                        
                except Exception as e:
                    pass 

            # ==========================================
            # KILL CHAIN UI: THE LAUNCH WINDOW
            # ==========================================
            if launch_auth:
                box_w = 360
                box_h = 60
                bx = int(w_frame/2 - box_w/2)
                by = 20
                
                if int(time.time() * 5) % 2 == 0:
                    cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), self.C_RED, -1)
                    text_color = (255, 255, 255)
                else:
                    cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (255, 255, 255), -1)
                    text_color = self.C_RED
                    
                cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), self.C_DARK, 2)
                cv2.putText(frame, "LAUNCH AUTHORIZED", (bx + 25, by + 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, text_color, 2)
                cv2.putText(frame, f"TTI: {tti:.3f} SEC", (bx + 110, by + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # --- TELEMETRY PANEL ---
        panel_w = 300
        panel_h = 260
        panel_x = w_frame - panel_w - 20
        panel_y = 20
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), self.C_DARK, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), self.C_GRAY, 1)
        
        cv2.putText(frame, "3D SPATIAL ENGINE", (panel_x + 15, panel_y + 25), cv2.FONT_HERSHEY_PLAIN, 1.2, self.C_CYAN, 2)
        cv2.line(frame, (panel_x + 10, panel_y + 35), (panel_x + panel_w - 10, panel_y + 35), self.C_GRAY, 1)
        
        cv2.putText(frame, f"SYS: {status}", (panel_x + 15, panel_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"PING: {self.current_dt*1000:.1f} ms", (panel_x + 15, panel_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,255,100), 1)
        
        cv2.putText(frame, f"3D VEL: {speed_3d:.2f} m/s", (panel_x + 15, panel_y + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_CYAN, 1)
        
        if "LOCKED" in status:
            cv2.putText(frame, "CURRENT POS (METERS)", (panel_x + 15, panel_y + 140), cv2.FONT_HERSHEY_PLAIN, 1.0, self.C_GRAY, 1)
            cv2.putText(frame, f"X: {curr_3d[0]:+.2f}m", (panel_x + 15, panel_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_CYAN, 1)
            cv2.putText(frame, f"Y: {curr_3d[1]:+.2f}m", (panel_x + 115, panel_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_CYAN, 1)
            cv2.putText(frame, f"Z: {curr_3d[2]:+.2f}m", (panel_x + 215, panel_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_CYAN, 1)
            
            cv2.putText(frame, "FUTURE POS (METERS)", (panel_x + 15, panel_y + 195), cv2.FONT_HERSHEY_PLAIN, 1.0, self.C_GRAY, 1)
            cv2.putText(frame, f"X: {fut_3d[0]:+.2f}m", (panel_x + 15, panel_y + 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_AMBER, 1)
            cv2.putText(frame, f"Y: {fut_3d[1]:+.2f}m", (panel_x + 115, panel_y + 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_AMBER, 1)
            cv2.putText(frame, f"Z: {fut_3d[2]:+.2f}m", (panel_x + 215, panel_y + 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_AMBER, 1)

        return frame