import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
from re import DEBUG, sub
from flask import Flask, send_from_directory, request, send_file, Response, jsonify, current_app
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob
from ultralytics import YOLO
import json
import base64

# Configure static folder properly
app = Flask(__name__)
app.static_folder = 'runs'
app.static_url_path = '/runs'
CORS(app)  # Enable CORS for all routes

# Ensure required directories exist with full permissions
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, mode=0o777, exist_ok=True)
    else:
        os.chmod(directory, 0o777)

# Clean up and recreate directories
def setup_directories():
    # Remove the entire runs directory if it exists
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    
    # Create fresh directories
    ensure_dir('runs')
    ensure_dir('runs/detect')
    ensure_dir('runs/detect/model1')
    ensure_dir('runs/detect/model2')
    ensure_dir('uploads')

# Initial setup
setup_directories()

# Initialize both YOLO models
try:
    model1 = YOLO('best_Hazard_Detection.pt')
    model2 = YOLO('best_Crane_Defects.pt')
except Exception as e:
    print(f"Error loading YOLO models: {str(e)}")
    raise

@app.route("/")
def hello_world():
    return jsonify({"message": "API is running"})

@app.route("/upload", methods=["POST"])
def predict_img():
    try:
        if 'file' not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "success": False
            }), 400

        f = request.files['file']
        if f.filename == '':
            return jsonify({
                "error": "No file selected",
                "success": False
            }), 400
        
        # Secure the filename and save the file
        filename = secure_filename(f.filename)
        filepath = os.path.join('uploads', filename)
        f.save(filepath)
        print(f"File saved to: {filepath}")
        
        file_extension = filename.rsplit('.', 1)[1].lower() 
        
        if file_extension in ['jpg', 'jpeg', 'png']:
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({
                    "error": f"Failed to read image: {filepath}",
                    "success": False
                }), 400
            
            print(f"Successfully read image: {filepath}")
            
            # Clean up previous results
            setup_directories()
            print("Cleared previous results")
            
            try:
                # Process with first model
                print("Processing with model1...")
                detections1 = model1(img)  # Don't save here
                result_img1 = detections1[0].plot()
                output_path1 = os.path.join('runs/detect/model1', 'result.jpg')
                cv2.imwrite(output_path1, result_img1)
                print(f"Saved model1 result to: {output_path1}")
                
                # Process with second model
                print("Processing with model2...")
                detections2 = model2(img)  # Don't save here
                result_img2 = detections2[0].plot()
                output_path2 = os.path.join('runs/detect/model2', 'result.jpg')
                cv2.imwrite(output_path2, result_img2)
                print(f"Saved model2 result to: {output_path2}")
                
            except Exception as e:
                print(f"Error during model processing: {str(e)}")
                return jsonify({
                    "error": f"Model processing failed: {str(e)}",
                    "success": False
                }), 500
            
            # Verify the output files exist
            if not os.path.exists(output_path1) or not os.path.exists(output_path2):
                return jsonify({
                    "error": "Failed to save processed images",
                    "success": False
                }), 500
            
            return jsonify({
                "model1_image_path": "detect/model1/result.jpg",
                "model2_image_path": "detect/model2/result.jpg",
                "success": True
            })
        
        elif file_extension == 'mp4': 
            video_path = filepath
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return jsonify({
                    "error": "Failed to read video",
                    "success": False
                }), 400

            # get video dimensions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
            # Define the codec and create VideoWriter objects for both models
            # Use avc1 codec instead of mp4v for better web compatibility
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Changed from 'mp4v' to 'avc1'
            out1_path = 'runs/detect/model1/output.mp4'
            out2_path = 'runs/detect/model2/output.mp4'
            
            out1 = cv2.VideoWriter(out1_path, fourcc, 30.0, (frame_width, frame_height))
            out2 = cv2.VideoWriter(out2_path, fourcc, 30.0, (frame_width, frame_height))
            
            try:
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    print(f"Processing frame {frame_count}")

                    # Process with first model
                    results1 = model1(frame)
                    res_plotted1 = results1[0].plot()
                    out1.write(res_plotted1)

                    # Process with second model
                    results2 = model2(frame)
                    res_plotted2 = results2[0].plot()
                    out2.write(res_plotted2)

                    if frame_count % 10 == 0:  # Print progress every 10 frames
                        print(f"Processed {frame_count} frames")

                    if cv2.waitKey(1) == ord('q'):
                        break
                
                print(f"Finished processing {frame_count} frames")
            except Exception as e:
                print(f"Error processing video: {str(e)}")
                return jsonify({
                    "error": f"Video processing failed: {str(e)}",
                    "success": False
                }), 500
            finally:
                cap.release()
                out1.release()
                out2.release()

            # Verify output files exist and have content
            if not os.path.exists(out1_path) or not os.path.exists(out2_path):
                return jsonify({
                    "error": "Failed to save processed videos",
                    "success": False
                }), 500

            # Check file sizes
            size1 = os.path.getsize(out1_path)
            size2 = os.path.getsize(out2_path)
            print(f"Video 1 size: {size1} bytes")
            print(f"Video 2 size: {size2} bytes")

            if size1 == 0 or size2 == 0:
                return jsonify({
                    "error": "Generated videos are empty",
                    "success": False
                }), 500

            # Verify the videos are readable
            test_cap1 = cv2.VideoCapture(out1_path)
            test_cap2 = cv2.VideoCapture(out2_path)
            
            if not test_cap1.isOpened() or not test_cap2.isOpened():
                return jsonify({
                    "error": "Generated videos are not readable",
                    "success": False
                }), 500
                
            test_cap1.release()
            test_cap2.release()

            return jsonify({
                "model1_video_path": "detect/model1/output.mp4",
                "model2_video_path": "detect/model2/output.mp4",
                "success": True
            })
        
        return jsonify({
            "error": "Unsupported file type",
            "success": False
        }), 400

    except Exception as e:
        print(f"Error in predict_img: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route('/runs/<path:filename>')
def serve_static(filename):
    try:
        # Get the absolute path to the runs directory
        runs_dir = os.path.abspath('runs')
        return send_from_directory(
            directory=runs_dir,
            path=filename,
            as_attachment=False
        )
    except Exception as e:
        print(f"Error serving static file {filename}: {str(e)}")
        return jsonify({"error": f"File not found: {filename}"}), 404

def get_frame(video_path):
    try:
        video = cv2.VideoCapture(video_path)
        while True:
            success, image = video.read()
            if not success:
                break
            ret, jpeg = cv2.imencode('.jpg', image) 
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
            time.sleep(0.1)
    except Exception as e:
        print(f"Error in get_frame: {str(e)}")

@app.route("/video_feed/<model>")
def video_feed(model):
    try:
        video_path = f'runs/detect/{model}/output.mp4'
        if not os.path.exists(video_path):
            return jsonify({"error": "Video not found"}), 404
        return Response(get_frame(video_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/runs/<path:filename>')
def serve_detected_image(filename):
    return send_from_directory('runs/', filename)


@app.route('/stream', methods=['GET'])
def video_stream():
    def generate_frames():
        # Initialize video capture from your source (e.g., RTSP stream, webcam, etc.)
        cap = None
        try:
            # Try to open webcam first
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Failed to open webcam, trying RTSP stream...")
                # Replace this URL with your RTSP stream URL
                cap = cv2.VideoCapture('rtsp://your_rtsp_url_here')
                if not cap.isOpened():
                    print("Failed to open video source")
                    yield f"data: {json.dumps({'error': 'Failed to open video source'})}\n\n"
                    return

            print("Successfully opened video source")
            
            while True:
                try:
                    success, frame = cap.read()
                    if not success:
                        print("Failed to read frame")
                        yield f"data: {json.dumps({'error': 'Failed to read frame'})}\n\n"
                        break
                    
                    try:
                        # Process frame with both models
                        results1 = model1(frame)
                        results2 = model2(frame)
                        
                        # Get processed frames
                        processed_frame1 = results1[0].plot()
                        processed_frame2 = results2[0].plot()
                        
                        # Encode frames to base64
                        _, buffer1 = cv2.imencode('.jpg', processed_frame1)
                        _, buffer2 = cv2.imencode('.jpg', processed_frame2)
                        
                        # Convert to base64
                        base64_frame1 = base64.b64encode(buffer1).decode('utf-8')
                        base64_frame2 = base64.b64encode(buffer2).decode('utf-8')
                        
                        # Create frame data
                        frame_data = {
                            'success': True,
                            'model1_frame': base64_frame1,
                            'model2_frame': base64_frame2
                        }
                        
                        # Convert to JSON and yield
                        yield f"data: {json.dumps(frame_data)}\n\n"
                        
                    except Exception as e:
                        print(f"Error processing frame: {str(e)}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        continue
                    
                    # Add a small delay to control frame rate
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error in frame loop: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    break
                    
        except Exception as e:
            print(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if cap is not None:
                print("Releasing video capture")
                cap.release()
    
    response = Response(generate_frames(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['X-Accel-Buffering'] = 'no'
    return response
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov9 models")
    parser.add_argument("--port", default=5001, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port) 