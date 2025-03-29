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
# UG
from collections import defaultdict
import requests 
import json
# UG
from depth_map_model import PhysicsGuidedDepthEstimation,LightweightFeatureExtractor, SimpleAttention, SparseDepthEstimator, DepthCompletion
import sys
import matplotlib.pyplot as plt
# from torchvision.transforms import Compose, ToTensor, Normalize
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import re

# Initialize the depth estimation model
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    depth_model = AutoModelForDepthEstimation.from_pretrained(model_name)
    torch.set_float32_matmul_precision('medium')  # Optimize matrix operations
    depth_model.to(device)  # Move model to GPU if available
    depth_model.eval()  # Set model to evaluation mode
except Exception as e:
    print(f"Error loading depth map model: {str(e)}")
    raise

# from midas.model_loader import load_model

# Configure static folder properly
app = Flask(__name__)
app.static_folder = 'runs'
app.static_url_path = '/runs'
CORS(app)  # Enable CORS for all routes

sys.modules['__main__'].PhysicsGuidedDepthEstimation = PhysicsGuidedDepthEstimation  # Register the class manually
sys.modules['__main__'].LightweightFeatureExtractor = LightweightFeatureExtractor
sys.modules['__main__'].SimpleAttention = SimpleAttention
sys.modules['__main__'].SparseDepthEstimator = SparseDepthEstimator
sys.modules['__main__'].DepthCompletion = DepthCompletion

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
    ensure_dir('runs/detect/depth_model')
    ensure_dir('uploads')

# Initial setup
setup_directories()

# Initialize both YOLO models
try:
    model1 = YOLO('best_Hazard_Detection.pt')
    model2 = YOLO('best_Crane_Defects.pt')
    # depth_model_type = "DPT_Large"
    # depth_model, transform, device = load_model(depth_model_type)
except Exception as e:
    print(f"Error loading YOLO models: {str(e)}")
    raise

def visualize_depth(depth_map, cmap='plasma'):
    # Avoid invalid values
    min_depth, max_depth = np.percentile(depth_map, [2, 98])  # Robust scaling
    depth_norm = np.clip(depth_map, min_depth, max_depth)
    depth_norm = (depth_norm - min_depth) / (max_depth - min_depth + 1e-8)  # Normalize

    return plt.cm.get_cmap(cmap)(depth_norm)[..., :3]  # Remove alpha channel

# Function to generate depth maps for test images
def generate_depth_maps(model, test_image, output_dir, img_height=192, img_width=256):
    """
    Generate depth maps for a directory of test images
    
    Args:
        model: Trained depth estimation model
        test_images_dir: Directory containing test images
        output_dir: Directory to save depth maps
        img_height, img_width: Height and width for resizing images
    """
    # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    
    # Save original dimensions for later resizing
    original_h, original_w = test_image.shape[:2]
    
    # Preprocess image: Resize, Normalize, Convert to Tensor
    image_resized = cv2.resize(test_image, (img_width, img_height))
    image_tensor = torch.from_numpy(image_resized).float() / 255.0  # Normalize to [0,1]
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)
    image_tensor = image_tensor.to(device)  # Move to GPU if available

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Ensure correct output format
    if isinstance(outputs, dict):
        depth_map = outputs['depth'].squeeze().cpu().numpy()  # Some models return dict output
    else:
        depth_map = outputs.squeeze().cpu().numpy()  # If model returns direct tensor
    
    # If model outputs log-depth, convert back to linear scale
    if np.min(depth_map) < 0:
        depth_map = np.exp(depth_map)

    # Resize depth map to original image dimensions
    depth_map_resized = cv2.resize(depth_map, (original_w, original_h))

    # Normalize depth map for visualization
    depth_colored = visualize_depth(depth_map_resized)
    
    # Save raw depth data (numpy format)
    np.save(os.path.join(output_dir, "result_depth.npy"), depth_map_resized)
    
    # Save colored visualization
    plt.imsave(os.path.join(output_dir, "result_depth_colored.png"), depth_colored)
    
    # Create side-by-side visualization
    # plt.figure(figsize=(12, 6))
    
    # plt.subplot(1, 2, 1)
    # plt.title("Input Image")
    # plt.imshow(test_image)
    # plt.axis('off')
    
    # plt.subplot(1, 2, 2)
    # plt.title("Depth Map")
    # plt.imshow(depth_colored)
    # plt.axis('off')
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "result_comparison.png"))
    # plt.close()
    return depth_colored

# Function to generate depth maps for test images
def generate_depth_maps_video(model, test_image, img_height=192, img_width=256):
    """
    Generate depth maps for a directory of test images
    
    Args:
        model: Trained depth estimation model
        test_images_dir: Directory containing test images
        output_dir: Directory to save depth maps
        img_height, img_width: Height and width for resizing images
    """
    # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    
    # Save original dimensions for later resizing
    original_h, original_w = test_image.shape[:2]
    
    # Preprocess image: Resize, Normalize, Convert to Tensor
    image_resized = cv2.resize(test_image, (img_width, img_height))
    image_tensor = torch.from_numpy(image_resized).float() / 255.0  # Normalize to [0,1]
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # Convert to (1, C, H, W)
    image_tensor = image_tensor.to(device)  # Move to GPU if available

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Ensure correct output format
    if isinstance(outputs, dict):
        depth_map = outputs['depth'].squeeze().cpu().numpy()  # Some models return dict output
    else:
        depth_map = outputs.squeeze().cpu().numpy()  # If model returns direct tensor
    
    # If model outputs log-depth, convert back to linear scale
    if np.min(depth_map) < 0:
        depth_map = np.exp(depth_map)

    # Resize depth map to original image dimensions
    depth_map_resized = cv2.resize(depth_map, (original_w, original_h))

    # Normalize depth map for visualization
    depth_colored = visualize_depth(depth_map_resized)
    
    # Create side-by-side visualization
    # plt.figure(figsize=(12, 6))
    
    # plt.subplot(1, 2, 1)
    # plt.title("Input Image")
    # plt.imshow(test_image)
    # plt.axis('off')
    
    # plt.subplot(1, 2, 2)
    # plt.title("Depth Map")
    # plt.imshow(depth_colored)
    # plt.axis('off')
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, "result_comparison.png"))
    # plt.close()
    return depth_colored

# Resize image to reduce computational load
def resize_image(image, max_width=640):
    h, w = image.shape[:2]
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
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
                output_path1 = os.path.join('runs\\detect\\model1', 'result.jpg')
                cv2.imwrite(output_path1, result_img1)
                print(f"Saved model1 result to: {output_path1}")
                
                # Process with second model
                print("Processing with model2...")
                detections2 = model2(img)  # Don't save here
                result_img2 = detections2[0].plot()
                output_path2 = os.path.join('runs\\detect\\model2', 'result.jpg')
                cv2.imwrite(output_path2, result_img2)
                print(f"Saved model2 result to: {output_path2}")
                
                # # # Process with depth map model
                # # # Convert BGR to RGB
                # # img_depth_map = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # # # Convert to Tensor (H, W, C) → (C, H, W) and normalize
                # # img_depth_map = torch.tensor(img_depth_map, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                # # img_depth_map = img_depth_map.to(device)

                # # with torch.no_grad():
                # #     depth_map = depth_model(img)
                # generate_depth_maps(
                #     model=depth_model,
                #     test_image=img,
                #     output_dir="runs\\detect\\depth_model",
                # )

                # # # Convert output to NumPy and normalize for visualization
                # # depth_map = depth_map.squeeze().cpu().numpy()
                # # depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                # # depth_map = depth_map.astype(np.uint8)
                
                # Load image and resize
                img = resize_image(img)

                if isinstance(img, np.ndarray):
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    print("hello")

                # Prepare the image for the model
                inputs = processor(images=img, return_tensors="pt").to(device)
                
                # Disable gradient calculation for inference
                with torch.amp.autocast("cuda"):  # Mixed precision inference
                    with torch.no_grad():
                        outputs = depth_model(**inputs)
                        predicted_depth = outputs.predicted_depth
                
                # Process the depth prediction
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(img.height, img.width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
                # Normalize the depth map
                output = prediction.cpu().numpy()
                depth_map = (output - output.min()) / (output.max() - output.min()) * 255
                depth_map = depth_map.astype(np.uint8)

                # Save depth map
                # cv2.imwrite(depth_output_path, depth_map)
                depth_output_path = os.path.join('runs\\detect\\depth_model', "result_depth_colored.png")
                cv2.imwrite(depth_output_path, depth_map)                
                print(f"Saved depth_model result to: {depth_output_path}")

                # Extract detected classes from both models
                detected_classes = []
                
                # Get classes from model1
                for detection in detections1:
                    for box in detection.boxes:
                        class_id = int(box.cls)
                        class_name = model1.names[class_id]
                        detected_classes.append(class_name)
                
                # Get classes from model2
                for detection in detections2:
                    for box in detection.boxes:
                        class_id = int(box.cls)
                        class_name = model2.names[class_id]
                        detected_classes.append(class_name)
                
                # Get overview from Groq API
                if detected_classes:  # Only call if we detected objects
                    overviews = get_object_overview_from_groq(detected_classes)
                    print("Generated overviews:", overviews)
                else:
                    overviews = {}
                    print("No detections found")
                
                # Log the overviews to verify data
                print("Object Descriptions:", overviews)
                
            except Exception as e:
                print(f"Error during model processing: {str(e)}")
                return jsonify({
                    "error": f"Model processing failed: {str(e)}",
                    "success": False
                }), 500
            
            # Verify the output files exist
            if not os.path.exists(output_path1) or not os.path.exists(output_path2) or not os.path.exists(depth_output_path):
                return jsonify({
                    "error": "Failed to save processed images",
                    "success": False
                }), 500
            
            return jsonify({
                "model1_image_path": "detect/model1/result.jpg",
                "model2_image_path": "detect/model2/result.jpg",
                "depth_image_path": "detect/depth_model/result_depth_colored.png",
                "object_descriptions": overviews,
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
            depth_output_path = 'runs/detect/depth_model/output.mp4'

            out1 = cv2.VideoWriter(out1_path, fourcc, 30.0, (frame_width, frame_height))
            out2 = cv2.VideoWriter(out2_path, fourcc, 30.0, (frame_width, frame_height))
            depth_out = cv2.VideoWriter(depth_output_path, fourcc, 20.0, (frame_width, frame_height), False)

            try:
                frame_count = 0
                detected_classes = []
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

                    # Get classes from model1
                    for detection in results1:
                        for box in detection.boxes:
                            class_id = int(box.cls)
                            class_name = model1.names[class_id]
                            detected_classes.append(class_name)
                    
                    # Get classes from model2
                    for detection in results2:
                        for box in detection.boxes:
                            class_id = int(box.cls)
                            class_name = model2.names[class_id]
                            detected_classes.append(class_name)

                    # # # Process frame with Depth Model
                    # # depth_video_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    # # depth_video_img = torch.tensor(depth_video_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                    # # depth_video_img = depth_video_img.to(device)

                    # # with torch.no_grad():
                    # #     depth_map = depth_model(depth_video_img)

                    # # # Convert depth map to NumPy and normalize for saving
                    # # depth_map = depth_map.squeeze().cpu().numpy()
                    # # depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                    # # depth_map = depth_map.astype(np.uint8)
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    # depth_map = generate_depth_maps_video(
                    #     model=depth_model,
                    #     test_image=frame,
                    # )

                    # # Ensure the depth map is in a supported format (CV_8U)
                    # if depth_map.dtype != np.uint8:
                    #     depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                    #     depth_map = depth_map.astype(np.uint8)

                    # Load image and resize
                    # frame = resize_image(frame)

                    if isinstance(frame, np.ndarray):
                        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        print("hello")

                    # Prepare the image for the model
                    inputs = processor(images=frame, return_tensors="pt").to(device)
                    
                    # Disable gradient calculation for inference
                    with torch.amp.autocast("cuda"):  # Mixed precision inference
                        with torch.no_grad():
                            outputs = depth_model(**inputs)
                            predicted_depth = outputs.predicted_depth
                    
                    # Process the depth prediction
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=(frame.height, frame.width),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                    
                    # Normalize the depth map
                    output = prediction.cpu().numpy()
                    depth_map = (output - output.min()) / (output.max() - output.min()) * 255
                    depth_map = depth_map.astype(np.uint8)

                    # Write depth frame to video
                    depth_out.write(depth_map)

                    if frame_count % 10 == 0:  # Print progress every 10 frames
                        print(f"Processed {frame_count} frames")

                    if cv2.waitKey(1) == ord('q'):
                        break
                
                print(f"Finished processing {frame_count} frames")
                # Get overview from Groq API
                # print(detected_classes)
                if detected_classes:  # Only call if we detected objects
                    overviews = get_object_overview_from_groq(detected_classes)
                    print("Generated overviews:", overviews)
                else:
                    overviews = {}
                    print("No detections found")
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
                depth_out.release()

            # Verify output files exist and have content
            if not os.path.exists(out1_path) or not os.path.exists(out2_path) or not os.path.exists(depth_output_path):
                return jsonify({
                    "error": "Failed to save processed videos",
                    "success": False
                }), 500

            # Check file sizes
            size1 = os.path.getsize(out1_path)
            size2 = os.path.getsize(out2_path)
            size3 = os.path.getsize(depth_output_path)
            print(f"Video 1 size: {size1} bytes")
            print(f"Video 2 size: {size2} bytes")
            print(f"Depth Video size: {size3} bytes")

            if size1 == 0 or size2 == 0 or size3 == 0:
                return jsonify({
                    "error": "Generated videos are empty",
                    "success": False
                }), 500

            # Verify the videos are readable
            test_cap1 = cv2.VideoCapture(out1_path)
            test_cap2 = cv2.VideoCapture(out2_path)
            test_cap3 = cv2.VideoCapture(depth_output_path)

            if not test_cap1.isOpened() or not test_cap2.isOpened():
                return jsonify({
                    "error": "Generated videos are not readable",
                    "success": False
                }), 500
                
            test_cap1.release()
            test_cap2.release()
            test_cap3.release()

            return jsonify({
                "model1_video_path": "detect/model1/output.mp4",
                "model2_video_path": "detect/model2/output.mp4",
                "depth_video_path": "detect/depth_model/output.mp4",
                "object_descriptions": overviews,
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

def get_object_overview_from_groq(detected_classes):
    """
    Send detected object classes to Groq API for descriptive overview.
    Returns a dictionary mapping class names to their overviews.
    """

    if not detected_classes:
        return {}

    class_counts = defaultdict(int)
    for class_name in detected_classes:
        class_counts[class_name] += 1

    detected_objects_str = ', '.join([f"{count} {cls}" for cls, count in class_counts.items()])

    # Replace with your actual API key
    
    api_key = "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"
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",  # Ensure this model exists
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides clear, concise object descriptions."
            },
            {
                "role": "user",
                "content": (
                    f"Based on the class detected: If class is Corrosion or Crack describe its effects and how harmful it can be. Also mention if cracks are there to not move as it may be risky. If it is corrosion tell to replace parts or make fixes. If class is 'hat' or 'vest' display a safety message. If it is 'no hat' or 'no vest' display a warning and it instruct worker to wear hat or vest as required.If any type of vehicle is detected display its type and its purpose and region it occupies also issue warnings to move the crane to move safely if such vehicles are present.Don't provide the same messages again and again. Try to be a bit different every time. If multiple objects of the same type are detected, mention that another similar object is being detected. Don't exactly copy from it . Try to be unique every time.\n"
                    f"{detected_objects_str}\n\n"
                    "Respond ONLY with a JSON object where each key is an object name "
                    "and each value is its description, like this:\n"
                    "{\n"
                    "  \"object1\": \"description1\",\n"
                    "  \"object2\": \"description2\"\n"
                    "}"
                )
            }
        ],
        "temperature": 0.3,
        "max_tokens": 512  # Lowered max tokens to prevent excessive response size
    }
    
    # print("\nSending request to Groq API with payload:")
    # print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raises error for non-200 responses
        
        content = response.json()
        
        # print("\nRaw API Response:")
        # print(json.dumps(content, indent=2))
        
        if "choices" in content and len(content["choices"]) > 0:
            overview_text = content["choices"][0]["message"]["content"]
            print("\nRaw API Response Content:", repr(overview_text))  # Debugging

            # Extract only the JSON part
            json_match = re.search(r'\{.*\}', overview_text, re.DOTALL)
            if json_match:
                overview_text = json_match.group(0)

            overview_text = overview_text.strip().encode("utf-8").decode("utf-8", "ignore")  # Remove unwanted characters

            return json.loads(overview_text)
        else:
            print("Unexpected response format:", content)
            return {}

    except requests.exceptions.RequestException as e:
        print(f"API Error: {str(e)}")
        if hasattr(e, 'response') and e.response:
            print(f"Response content: {e.response.text}")
        return {}

    except json.JSONDecodeError:
        print("Failed to decode JSON response")
        return {}

    
@app.route('/stream', methods=['GET'])
def video_stream():
    def generate_frames():
        # Initialize video capture from your source (e.g., RTSP stream, webcam, etc.)
        # Replace this URL with your video source
        cap = cv2.VideoCapture(0)  # Use 0 for webcam or RTSP URL for IP camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Failed to open video capture")
            cap = cv2.VideoCapture(0)  # Use 0 for webcam or RTSP URL for IP camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(cap.isOpened())
            if not cap.isOpened():
                yield f"data: {json.dumps({'error': 'Failed to open video capture'})}\n\n"
                return

        try:
            while True:
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
                    
                    # # Process frame with Depth Model
                    # depth_map = generate_depth_maps_video(
                    #     model=depth_model,
                    #     test_image=frame,
                    # )

                    # # Ensure the depth map is in a supported format (CV_8U)
                    # if depth_map.dtype != np.uint8:
                    #     depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
                    #     depth_map = depth_map.astype(np.uint8)
                
                    # Load image and resize
                    # frame = resize_image(frame)

                    if isinstance(frame, np.ndarray):
                        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        print("hello")

                    # Prepare the image for the model
                    inputs = processor(images=frame, return_tensors="pt").to(device)
                    
                    # Disable gradient calculation for inference
                    with torch.amp.autocast("cuda"):  # Mixed precision inference
                        with torch.no_grad():
                            outputs = depth_model(**inputs)
                            predicted_depth = outputs.predicted_depth
                    
                    # Process the depth prediction
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=(frame.height, frame.width),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                    
                    # Normalize the depth map
                    output = prediction.cpu().numpy()
                    depth_map = (output - output.min()) / (output.max() - output.min()) * 255
                    depth_map = depth_map.astype(np.uint8)

                    # Encode frames to base64
                    _, buffer1 = cv2.imencode('.jpg', processed_frame1)
                    _, buffer2 = cv2.imencode('.jpg', processed_frame2)
                    _, buffer_depth = cv2.imencode('.jpg', depth_map)

                    # Convert to base64
                    base64_frame1 = base64.b64encode(buffer1).decode('utf-8')
                    base64_frame2 = base64.b64encode(buffer2).decode('utf-8')
                    base64_depth_frame = base64.b64encode(buffer_depth).decode('utf-8')

                    # Create frame data
                    frame_data = {
                        'success': True,
                        'model1_frame': base64_frame1,
                        'model2_frame': base64_frame2,
                        'depth_frame': base64_depth_frame,
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
            print(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if cap is not None:
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