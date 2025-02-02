from flask import Flask, Response, render_template, jsonify, send_file
from flask_socketio import SocketIO
from camera_manager import CameraManager
import time
import threading
import json
import io

app = Flask(__name__, static_folder='static', static_url_path='/static')
socketio = SocketIO(app)

print("1. Starting script")

camera_manager = CameraManager()
camera_manager.initialize_cameras()

def send_camera_updates():
    while True:
        camera_data = camera_manager.get_camera_data()
        camera_data['isCalibration'] = False  # Regular update
        socketio.emit('camera_positions_update', camera_data)
        socketio.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html', 
                         num_cameras=camera_manager.num_cameras,
                         error_message=camera_manager.error_message,
                         using_mock=camera_manager.using_mock)

@app.route('/config')
def get_config():
    """Return current camera configuration"""
    config_data = {
        'camera_positions': camera_manager.camera_positions,
        'calibration_data': getattr(camera_manager, 'calibration_data', {}),
        'using_mock': camera_manager.using_mock  # Add mock status to config
    }
    print("Sending config data:", config_data)  # Debug print
    return jsonify(config_data)

@app.route('/placeholder_frame/<int:camera_id>')
def placeholder_frame(camera_id):
    frame_bytes = camera_manager.get_placeholder_frame(camera_id)
    if frame_bytes:
        return send_file(io.BytesIO(frame_bytes), mimetype='image/jpeg')
    else:
        return "Camera not available", 404

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

@socketio.on('toggle_camera_stream')
def toggle_camera_stream(data):
    action = data['action']
    if action == 'start':
        success = camera_manager.start_stream()
    else:
        success = camera_manager.stop_stream()
    socketio.emit('stream_toggle_response', {'success': success, 'action': action})

@socketio.on('toggle_dot_detection')
def toggle_dot_detection(data):
    success = camera_manager.toggle_dot_detection(data['enable'])
    socketio.emit('dot_detection_toggle_response', {'success': success, 'enabled': data['enable']})

@socketio.on('calibrate_cameras')
def handle_calibration():
    success, message, new_positions = camera_manager.calibrate_cameras()
    
    if success and new_positions:
        # Send camera positions update with calibration flag
        camera_data = camera_manager.get_camera_data()
        camera_data['isCalibration'] = True  # Add flag to indicate this is a calibration update
        socketio.emit('camera_positions_update', camera_data)
        
        # Send calibration response without alert
        socketio.emit('calibration_response', {
            'success': success,
            'message': message,
            'silent': True  # Flag to prevent alert
        })
    else:
        # Send failure response with alert
        socketio.emit('calibration_response', {
            'success': success,
            'message': message,
            'silent': False
        })

@socketio.on('connect')
def handle_connect():
    print("Client connected")

if __name__ == '__main__':
    try:
        print("4. Starting Flask app")
        print(f"Using {'mock' if camera_manager.using_mock else 'real'} cameras")
        
        # Start the camera update thread
        camera_update_thread = threading.Thread(target=send_camera_updates)
        camera_update_thread.daemon = True
        camera_update_thread.start()
        
        socketio.run(app, debug=False, port=3001)
        print("5. Flask app has finished running")
    except Exception as e:
        print(f"Error running application: {str(e)}")
    finally:
        camera_manager.close_cameras()

print("7. Script execution completed")