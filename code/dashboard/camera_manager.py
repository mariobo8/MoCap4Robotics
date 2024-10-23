import cv2
import numpy as np
from pseyepy import Camera
import time
import math
import os
import json
from datetime import datetime

class CameraManager:
    def __init__(self):
        self.cameras = None
        self.num_cameras = 3  # Set a default value
        self.error_message = None
        self.streaming = False
        self.placeholder_frames = []
        self.resolutions = []
        self.detect_dots = False
        self.camera_positions = []
        self.config_path = 'code/dashboard/config/camera_params.json'
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        # Load config first
        self.load_camera_config()        

    def initialize_cameras(self):
        try:
            print("Attempting to initialize cameras")
            self.cameras = Camera([0, 1, 2], fps=30, resolution=Camera.RES_LARGE, colour=True)
            print(f"Cameras initialized: fps={self.cameras.fps}, resolution={self.cameras.resolution}, colour={self.cameras.colour}")
            self.num_cameras = len(self.cameras.exposure)
            print(f"Number of cameras: {self.num_cameras}")
            self.resolutions = self.cameras.resolution
            
            # Only set default positions if none were loaded
            if not self.camera_positions:
                self.camera_positions = [[0, 0, 0] for _ in range(self.num_cameras)]
                
        except Exception as e:
            print(f"Error initializing cameras: {str(e)}")
            self.cameras = None
            self.num_cameras = 3
            self.error_message = str(e)
            self.resolutions = [(640, 480)] * self.num_cameras
            
            # Only set default positions if none were loaded
            if not self.camera_positions:
                self.set_default_positions()

        # Create placeholder frames
        self.placeholder_frames = []
        for i in range(self.num_cameras):
            width, height = self.resolutions[i]
            placeholder = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Camera {i+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.placeholder_frames.append(placeholder)

    def load_camera_config(self):
        """
        Load camera configuration from JSON file.
        If file doesn't exist or is invalid, use default values.
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load camera positions if they exist
                if 'camera_positions' in config:
                    self.camera_positions = config['camera_positions']
                    print("Loaded camera positions from config:", self.camera_positions)
                else:
                    self.set_default_positions()
                
                # Load other calibration data
                self.calibration_data = config.get('calibration_data', {})
                print("Loaded calibration data from config")
            else:
                print("No config file found, using default positions")
                self.set_default_positions()
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            self.set_default_positions()


    def detect_white_dots(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image to get white regions
        _, thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and circularity
        dots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 0.4 < area < 1000:  # Adjust these values based on the expected dot size
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.1:  # Adjust this threshold as needed
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        dots.append((cX, cY))
        
        return dots

    def mark_dots(self, frame, dots):
        for (x, y) in dots:
            cv2.drawMarker(frame, (x, y), (0, 0, 255), cv2.MARKER_STAR, 10, 3)
        return frame

    def process_frame(self, frame):
        if self.detect_dots:
            dots = self.detect_white_dots(frame)
            frame = self.mark_dots(frame, dots)
        return frame

    def gen_frames(self, camera_index):
        while True:
            if self.streaming and self.cameras:
                frame, timestamp = self.cameras.read(camera_index)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_bgr = self.process_frame(frame_bgr)
            else:
                frame_bgr = self.placeholder_frames[camera_index]
            
            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def update_camera_settings(self, exposure, gain):
        try:
            self.cameras.exposure = [exposure] * self.num_cameras
            self.cameras.gain = [gain] * self.num_cameras
            return True, None
        except Exception as e:
            return False, str(e)

    def start_stream(self):
        if self.cameras:
            self.streaming = True
            return True
        return False

    def stop_stream(self):
        self.streaming = False
        return True

    def toggle_dot_detection(self, enable):
        self.detect_dots = enable
        return True

    def close_cameras(self):
        if self.cameras:
            print("Closing cameras")
            self.cameras.end()

    
    def get_camera_positions(self):
        # TODO: Replace this with actual camera position retrieval logic
        # For now, we'll return the stored positions
        return self.camera_positions

    def update_camera_positions(self):
        # This method simulates camera movement
        # Replace this with actual position updates when available
        for i in range(self.num_cameras):
            angle = 0.5 + (2 * math.pi * i / self.num_cameras)
            self.camera_positions[i] = [
                2 * math.sin(angle),
                2,
                2 * math.cos(angle)
            ]


    def get_camera_data(self):
        """
        Returns the current camera positions and look-at points.
        """
        return {
            'positions': self.camera_positions,
            'lookAts': [[0, 0, 0]] * self.num_cameras,
            'timestamp': time.time()  # Add timestamp for frontend to detect updates
        }
    
    def calibrate_pair(self, pts1, pts2):
        """
        Calibrate a pair of cameras using the 8-point algorithm.
        Returns rotation matrix and translation vector.
        """
        # Normalize points to [-1, 1] range
        pts1_norm = pts1.copy()
        pts2_norm = pts2.copy()
        
        # Normalize x coordinates
        pts1_norm[:, 0] = (pts1[:, 0] - self.resolutions[0][0]/2) / (self.resolutions[0][0]/2)
        pts2_norm[:, 0] = (pts2[:, 0] - self.resolutions[0][0]/2) / (self.resolutions[0][0]/2)
        
        # Normalize y coordinates
        pts1_norm[:, 1] = (pts1[:, 1] - self.resolutions[0][1]/2) / (self.resolutions[0][1]/2)
        pts2_norm[:, 1] = (pts2[:, 1] - self.resolutions[0][1]/2) / (self.resolutions[0][1]/2)

        # Build the constraint matrix A
        A = np.zeros((len(pts1_norm), 9))
        for i in range(len(pts1_norm)):
            x1, y1 = pts1_norm[i]
            x2, y2 = pts2_norm[i]
            A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]

        # Solve for E using SVD
        U, S, Vt = np.linalg.svd(A)
        E = Vt[-1].reshape(3, 3)
        
        # Enforce Essential matrix properties
        U, S, Vt = np.linalg.svd(E)
        S = np.array([1, 1, 0])
        E = U @ np.diag(S) @ Vt
        
        # Recover rotation and translation
        U, _, Vt = np.linalg.svd(E)
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        
        # Two possible rotations
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        
        if np.linalg.det(R1) < 0: R1 = -R1
        if np.linalg.det(R2) < 0: R2 = -R2
        
        t = U[:, 2]
        
        # Choose correct solution
        possible_Rs = [R1, R2]
        possible_ts = [t, -t]
        
        best_num_front = 0
        best_R = None
        best_t = None
        
        for R in possible_Rs:
            for t in possible_ts:
                num_front = 0
                for i in range(len(pts1_norm)):
                    p1 = np.array([pts1_norm[i][0], pts1_norm[i][1], 1])
                    p2 = np.array([pts2_norm[i][0], pts2_norm[i][1], 1])
                    
                    if p1[2] > 0 and (R @ p1 + t)[2] > 0:
                        num_front += 1
                
                if num_front > best_num_front:
                    best_num_front = num_front
                    best_R = R
                    best_t = t
        
        return best_R, best_t

    def calibrate_cameras(self):
        """
        Calibrate all three cameras and save the configuration.
        """
        try:
            if not self.streaming:
                return False, "Cameras must be streaming to perform calibration", None
                
            print("Starting 8-point calibration for all three cameras...")
            
            # 1. Set fixed position for camera 1
            camera1_pos = np.array([1.5, 1, -1])
            
            # 2. Get frames and detect dots for all cameras
            frames = []
            dots = []
            for i in range(3):
                frame, _ = self.cameras.read(i)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_dots = self.detect_white_dots(frame_bgr)
                frames.append(frame_bgr)
                dots.append(frame_dots)
                print(f"Camera {i+1} detected {len(frame_dots)} dots")
            
            # 3. Check minimum points requirement
            for i, camera_dots in enumerate(dots):
                if len(camera_dots) < 8:
                    return False, f"Camera {i+1} has insufficient points ({len(camera_dots)}). Need at least 8.", None
            
            # 4. Calibrate camera pairs
            # First pair (1-2)
            R12, t12 = self.calibrate_pair(np.float32(dots[0]), np.float32(dots[1]))
            # Store the transformation matrices
            self.R12 = R12
            self.t12 = t12
            
            # Second pair (2-3)
            R23, t23 = self.calibrate_pair(np.float32(dots[1]), np.float32(dots[2]))
            # Store the transformation matrices
            self.R23 = R23
            self.t23 = t23
            
            # 5. Calculate camera positions
            scale = 2.0  # Scale factor for reasonable distances
            
            # Camera 2 relative to Camera 1
            camera2_pos = camera1_pos + scale * t12
            
            # Camera 3 relative to Camera 2
            t23_global = camera2_pos + scale * (R12 @ t23)
            camera3_pos = t23_global
            
            # 6. Update camera positions
            self.camera_positions[0] = camera1_pos.tolist()
            self.camera_positions[1] = camera2_pos.tolist()
            self.camera_positions[2] = camera3_pos.tolist()
            
            print("\nCamera positions after calibration:")
            print("Camera 1:", self.camera_positions[0])
            print("Camera 2:", self.camera_positions[1])
            print("Camera 3:", self.camera_positions[2])
            
            # 7. Save the configuration
            if self.save_camera_config():
                print("Calibration configuration saved successfully")
            else:
                print("Warning: Failed to save calibration configuration")
            
            return True, "Three-camera calibration completed successfully", self.camera_positions
            
        except Exception as e:
            error_msg = f"Calibration failed: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return False, error_msg, None


    def load_camera_config(self):
        """
        Load camera configuration from JSON file.
        If file doesn't exist or is invalid, use default values.
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load camera positions
                if 'camera_positions' in config:
                    self.camera_positions = config['camera_positions']
                    print("Loaded camera positions from config:", self.camera_positions)
                else:
                    self.set_default_positions()
                
                # Load calibration data including transformation matrices
                if 'calibration_data' in config:
                    calib_data = config['calibration_data']
                    if calib_data.get('R12') and calib_data.get('t12'):
                        self.R12 = np.array(calib_data['R12'])
                        self.t12 = np.array(calib_data['t12'])
                    if calib_data.get('R23') and calib_data.get('t23'):
                        self.R23 = np.array(calib_data['R23'])
                        self.t23 = np.array(calib_data['t23'])
                    print("Loaded calibration matrices from config")
            else:
                print("No config file found, using default positions")
                self.set_default_positions()
                
            # Ensure camera_positions is initialized
            if not self.camera_positions:
                self.set_default_positions()
                
            return True
                
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            self.set_default_positions()
            return False

    def set_default_positions(self):
        """Set default camera positions"""
        self.camera_positions = [
            [1.5, 1, -1],      # Camera 1 default position
            [-1, 0, 1.73],     # Camera 2 default position
            [-1, 0, -1.73]     # Camera 3 default position
        ]
            
    def save_camera_config(self):
        """
        Save current camera configuration to JSON file.
        Includes camera positions and calibration metadata.
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            config = {
                'camera_positions': self.camera_positions,
                'calibration_data': {
                    'timestamp': datetime.now().isoformat(),
                    'num_cameras': self.num_cameras,
                    'resolution': self.resolutions,
                    'R12': self.R12.tolist() if hasattr(self, 'R12') else None,
                    't12': self.t12.tolist() if hasattr(self, 't12') else None,
                    'R23': self.R23.tolist() if hasattr(self, 'R23') else None,
                    't23': self.t23.tolist() if hasattr(self, 't23') else None
                }
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            print("Saved camera configuration to", self.config_path)
            return True
        except Exception as e:
            print(f"Error saving config: {str(e)}")
            traceback.print_exc()  # This will print the full error traceback
            return False