import cv2
import time
import os
import numpy as np 
from fusion_optics import AsymmetricFusionOptics  
from brain import AdaptiveLIFNeuron
from omega_brain import OmegaKinematicEngine
from interface import TacticalHUD

def boot_sequence():
    print("\n===================================================")
    print(" [COMMANDER] Booting Vajra-Drishti Master System ")
    print("===================================================")
    print(" [1] Live Camera Array")
    print(" [2] Recorded Video Telemetry")
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == '1': return 0, 1  
        elif choice == '2':
            video_path = input("Enter video file name or drag-and-drop here: ").strip()
            if video_path.startswith("& "): video_path = video_path[2:]
            return video_path.strip().strip('"').strip("'"), 30  

if __name__ == "__main__":
    sensor_source, loop_delay = boot_sequence()
    
    sensor = AsymmetricFusionOptics(camera_idx=sensor_source)
    neuron = AdaptiveLIFNeuron(leak_rate=0.85, base_threshold=100.0)
    tracker = OmegaKinematicEngine() 
    hud = TacticalHUD()
    
    print("[SYSTEM] ALL SYSTEMS GO. Entering Omni-Tactical Loop.\n")
    
    # --- SHADOW MODE STATE ---
    was_tracking = False

    while True:
        live_feed, event_feed, targets = sensor.scan_airspace()
        
        if live_feed is None:
            print("[SYSTEM] Video stream ended. Flushing Shadow Mode Cache...")
            hud.export_omega_dataset()
            break

        h_frame, w_frame = live_feed.shape[:2]

        target_x, target_y = 0, 0
        last_w, last_h = 0, 0
        target_area = 0
        status = "SEARCHING"
        
        if len(targets) > 0:
            primary = targets[0]
            target_x, target_y = primary['x'], primary['y']
            last_w, last_h, target_area = primary['w'], primary['h'], primary['area']
            status = "LOCKED"
            
        spiked = neuron.process_stimulus(target_area)
        
        # --- The Pure Omega Brain ---
        future_path, curr_3d, fut_3d, speed_3d, tti, launch_auth, omega_status = tracker.update(
            target_x, target_y, last_w, w_frame, h_frame, spiked
        )
        
        if status == "LOCKED":
            status = omega_status 
            
        # --- SHADOW MODE LOGGER ---
        hud.log_spacetime_telemetry(target_x, target_y, last_w, last_h, status)
        
        is_tracking = ("LOCKED" in status) or ("COASTING" in status)
        
        # Trigger: We just lost the lock. Dump the cache to a file.
        if was_tracking and not is_tracking:
            print("[SHADOW MODE] Target lost. Archiving flight path to vault...")
            hud.export_omega_dataset()
            
        was_tracking = is_tracking
        
        # --- Base HUD Render ---
        final_ui = hud.render(
            live_feed, target_x, target_y, last_w, last_h, 
            future_path, curr_3d, fut_3d, speed_3d, tti, launch_auth, status
        )
        
        cv2.imshow("Vajra-Drishti: Tactical Command", final_ui)
        cv2.imshow("Vajra-Drishti: Neuromorphic Event Stream", event_feed)
        
        key = cv2.waitKey(loop_delay) & 0xFF
        if key == 27: # ESC key
            print("[SYSTEM] Manual Override. Flushing Shadow Mode Cache...")
            hud.export_omega_dataset()
            break

    sensor.cap.release()
    cv2.destroyAllWindows()
    print("[SYSTEM] Power down complete.")