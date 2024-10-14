from flask import Flask, Response, render_template
from flask_socketio import SocketIO
from pseyepy import Camera
import cv2
import time

app = Flask(__name__)
socketio = SocketIO(app)

print("1. Starting script")

# Initialize cameras
try:
    print("2. Attempting to initialize cameras")
    cameras = Camera([0, 1, 2], fps=30, resolution=Camera.RES_LARGE, colour=True)
    print(f"3. Cameras initialized: fps={cameras.fps}, resolution={cameras.resolution}, colour={cameras.colour}")
    num_cameras = len(cameras.exposure)
    print(f"Number of cameras: {num_cameras}")
except Exception as e:
    print(f"Error initializing cameras: {str(e)}")
    cameras = None
    num_cameras = 0
    error_message = str(e)

def gen_frames(camera_index):
    while True:
        frame, timestamp = cameras.read(camera_index)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', frame_bgr)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1/30)  # Adjust as needed to match camera FPS

@app.route('/')
def index():
    return render_template('index.html', num_cameras=num_cameras, error_message=error_message if 'error_message' in locals() else None)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    if camera_id < num_cameras:
        return Response(gen_frames(camera_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Camera not available", 404

@socketio.on('update_camera_settings')
def update_camera_settings(data):
    try:
        cameras.exposure = [data['exposure']] * num_cameras
        cameras.gain = [data['gain']] * num_cameras
        socketio.emit('settings_updated', {'exposure': data['exposure'], 'gain': data['gain']})
    except Exception as e:
        socketio.emit('settings_update_failed', {'message': str(e)})

if __name__ == '__main__':
    try:
        print("4. Starting Flask app")
        socketio.run(app, debug=False, port=3001)
        print("5. Flask app has finished running")
    except Exception as e:
        print(f"Error running application: {str(e)}")
    finally:
        if cameras:
            print("6. Closing cameras")
            cameras.end()

print("7. Script execution completed")