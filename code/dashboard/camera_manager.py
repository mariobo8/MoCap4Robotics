import cv2
from pseyepy import Camera

class CameraManager:
    def __init__(self):
        self.cameras = None
        self.num_cameras = 0
        self.error_message = None

    def initialize_cameras(self):
        try:
            print("Attempting to initialize cameras")
            self.cameras = Camera([0, 1, 2], fps=30, resolution=Camera.RES_LARGE, colour=True)
            print(f"Cameras initialized: fps={self.cameras.fps}, resolution={self.cameras.resolution}, colour={self.cameras.colour}")
            self.num_cameras = len(self.cameras.exposure)
            print(f"Number of cameras: {self.num_cameras}")
        except Exception as e:
            print(f"Error initializing cameras: {str(e)}")
            self.cameras = None
            self.num_cameras = 0
            self.error_message = str(e)

    def gen_frames(self, camera_index):
        while True:
            frame, timestamp = self.cameras.read(camera_index)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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

    def close_cameras(self):
        if self.cameras:
            print("Closing cameras")
            self.cameras.end()