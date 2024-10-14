from flask import Flask, Response, render_template
from flask_socketio import SocketIO
from camera_manager import CameraManager
import time

app = Flask(__name__)
socketio = SocketIO(app)

print("1. Starting script")

camera_manager = CameraManager()
camera_manager.initialize_cameras()

@app.route('/')
def index():
    return render_template('index.html', num_cameras=camera_manager.num_cameras, error_message=camera_manager.error_message)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    if camera_id < camera_manager.num_cameras:
        return Response(camera_manager.gen_frames(camera_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Camera not available", 404

@socketio.on('update_camera_settings')
def update_camera_settings(data):
    success, error = camera_manager.update_camera_settings(data['exposure'], data['gain'])
    if success:
        socketio.emit('settings_updated', {'exposure': data['exposure'], 'gain': data['gain']})
    else:
        socketio.emit('settings_update_failed', {'message': error})

if __name__ == '__main__':
    try:
        print("4. Starting Flask app")
        socketio.run(app, debug=False, port=3001)
        print("5. Flask app has finished running")
    except Exception as e:
        print(f"Error running application: {str(e)}")
    finally:
        camera_manager.close_cameras()

print("7. Script execution completed")