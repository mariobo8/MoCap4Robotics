import cv2
import numpy as np
from pseyepy import Camera
import time
import math

class CameraManager:
    def __init__(self):
        self.cameras = None
        self.num_cameras = 0
        self.error_message = None
        self.streaming = False
        self.placeholder_frames = []
        self.resolutions = []
        self.detect_dots = False  # Flag to enable/disable dot detection
        self.camera_positions = []  # To store camera positions

    def initialize_cameras(self):
        try:
            print("Attempting to initialize cameras")
            self.cameras = Camera([0, 1, 2], fps=30, resolution=Camera.RES_LARGE, colour=True)
            print(f"Cameras initialized: fps={self.cameras.fps}, resolution={self.cameras.resolution}, colour={self.cameras.colour}")
            self.num_cameras = len(self.cameras.exposure)
            print(f"Number of cameras: {self.num_cameras}")
            self.resolutions = self.cameras.resolution
            # Initialize camera positions
            self.camera_positions = [[0, 0, 0] for _ in range(self.num_cameras)]
        except Exception as e:
            print(f"Error initializing cameras: {str(e)}")
            self.cameras = None
            self.num_cameras = 3
            self.error_message = str(e)
            self.resolutions = [(640, 480)] * self.num_cameras

        # Create placeholder (black) frames
        for i in range(self.num_cameras):
            width, height = self.resolutions[i]
            placeholder = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Camera {i+1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.placeholder_frames.append(placeholder)

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
            if 10 < area < 1000:  # Adjust these values based on the expected dot size
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.6:  # Adjust this threshold as needed
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
        self.update_camera_positions()  # Update positions (simulated movement)
        return {
            'positions': self.get_camera_positions(),
            'lookAts': [[0, 0, 0]] * self.num_cameras  # All cameras looking at origin
        }