"""
Flask Web Application for CycleGAN Face Photo ↔ Sketch Translation
====================================================================
A user-friendly web interface that:
  1. Allows users to upload a picture (photo or sketch).
  2. Automatically detects whether the input is a sketch or real face.
  3. Converts it accordingly using the trained CycleGAN model.
  4. Optionally supports live camera input.

Usage:
  python app.py
  Then open http://localhost:5000 in your browser.
"""

import os
import io
import base64
import uuid

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils

from flask import Flask, render_template, request, jsonify, send_from_directory

from cyclegan import Generator, Config, get_transforms as get_model_transforms

# ============================================================================
# Flask App Setup
# ============================================================================

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configuration
config = Config()
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Global model references (loaded once at startup)
G_AB = None  # Photo → Sketch
G_BA = None  # Sketch → Photo


def load_models():
    """Load the trained CycleGAN generators."""
    global G_AB, G_BA
    
    # Try multiple possible checkpoint locations
    checkpoint_paths = [
        "latest.pth",  # In Q1 folder (root)
        os.path.join(config.checkpoint_dir, "latest.pth"),  # In checkpoints/ subfolder
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    G_AB = Generator(n_residual_blocks=config.n_residual_blocks, ngf=config.ngf).to(config.device)
    G_BA = Generator(n_residual_blocks=config.n_residual_blocks, ngf=config.ngf).to(config.device)
    
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=config.device)
        G_AB.load_state_dict(ckpt["G_AB"])
        G_BA.load_state_dict(ckpt["G_BA"])
        print(f"[INFO] Models loaded from: {checkpoint_path}")
    else:
        print(f"[WARNING] No checkpoint found at {checkpoint_paths}. Using random weights.")
    
    G_AB.eval()
    G_BA.eval()


def detect_image_type(img_np):
    """
    Detect whether an image is a sketch or a real photo.
    
    Uses color saturation and edge density as heuristics:
    - Sketches tend to be grayscale or near-grayscale (low saturation).
    - Sketches have high contrast with white backgrounds.
    
    Args:
        img_np: NumPy array of the image (RGB, uint8).
    
    Returns:
        str: "sketch" or "photo"
    """
    # Convert to float for analysis
    img_float = img_np.astype(np.float32)
    
    # Check color saturation (std of channel differences)
    r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    rg_diff = np.std(r - g)
    rb_diff = np.std(r - b)
    gb_diff = np.std(g - b)
    avg_saturation = (rg_diff + rb_diff + gb_diff) / 3.0
    
    # Check brightness distribution (sketches tend to have bright backgrounds)
    brightness = np.mean(img_float)
    
    # Heuristic: low saturation → sketch
    if avg_saturation < 10 or (avg_saturation < 20 and brightness > 180):
        return "sketch"
    else:
        return "photo"


def translate_image(img_pil, direction=None):
    """
    Translate an image using the CycleGAN model.
    
    Args:
        img_pil: PIL Image (RGB).
        direction: "photo_to_sketch", "sketch_to_photo", or None (auto-detect).
    
    Returns:
        tuple: (output PIL Image, detected direction string)
    """
    # Auto-detect direction if not specified
    img_np = np.array(img_pil)
    
    if direction is None:
        detected = detect_image_type(img_np)
        direction = "sketch_to_photo" if detected == "sketch" else "photo_to_sketch"
    
    # Prepare input tensor
    transform = get_model_transforms(config.img_size, is_train=False)
    input_tensor = transform(img_pil).unsqueeze(0).to(config.device)
    
    # Run through appropriate generator
    with torch.no_grad():
        if direction == "photo_to_sketch":
            output_tensor = G_AB(input_tensor)
        else:
            output_tensor = G_BA(input_tensor)
    
    # Convert output tensor to PIL Image
    output_tensor = (output_tensor.squeeze(0).cpu() + 1) / 2.0  # Denormalize
    output_tensor = output_tensor.clamp(0, 1)
    output_pil = transforms.ToPILImage()(output_tensor)
    
    return output_pil, direction


# ============================================================================
# Routes
# ============================================================================

@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    """
    API endpoint for image translation.
    
    Accepts either:
    - An uploaded file (multipart form)
    - A base64-encoded image (JSON, for camera capture)
    
    Returns JSON with the translated image as base64 and metadata.
    """
    try:
        # Get direction from form data or JSON
        direction = request.form.get("direction", None)
        if direction is None and request.is_json:
            direction = request.json.get("direction", None)
        
        # Handle file upload
        if "image" in request.files:
            file = request.files["image"]
            img = Image.open(file.stream).convert("RGB")
        # Handle base64 image (from camera)
        elif request.is_json and "image_data" in request.json:
            img_data = request.json["image_data"]
            # Remove data URI prefix if present
            if "," in img_data:
                img_data = img_data.split(",")[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Auto-detect or use specified direction
        if direction not in ["photo_to_sketch", "sketch_to_photo"]:
            direction = None
        
        # Translate
        output_img, used_direction = translate_image(img, direction)
        
        # Convert output to base64
        buffered = io.BytesIO()
        output_img.save(buffered, format="PNG")
        output_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Also convert input to base64 for display
        input_buffered = io.BytesIO()
        img.save(input_buffered, format="PNG")
        input_b64 = base64.b64encode(input_buffered.getvalue()).decode("utf-8")
        
        return jsonify({
            "success": True,
            "input_image": f"data:image/png;base64,{input_b64}",
            "output_image": f"data:image/png;base64,{output_b64}",
            "direction": used_direction,
            "direction_label": "Photo → Sketch" if used_direction == "photo_to_sketch" else "Sketch → Photo"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "models_loaded": G_AB is not None and G_BA is not None,
        "device": str(config.device)
    })


# ============================================================================
# HTML Template (embedded for simplicity)
# ============================================================================

def create_templates():
    """Create the HTML template for the web UI."""
    os.makedirs("templates", exist_ok=True)
    
    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CycleGAN - Face Photo ↔ Sketch Translation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #fff;
            min-height: 100vh;
        }
        .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
        
        header {
            text-align: center;
            padding: 40px 0 30px;
        }
        header h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #f093fb, #f5576c, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        header p { color: #aaa; font-size: 1.1em; }
        
        .card {
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
        }
        
        .upload-area {
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            border-color: #4facfe;
            background: rgba(79, 172, 254, 0.05);
        }
        .upload-area.dragover {
            border-color: #f5576c;
            background: rgba(245, 87, 108, 0.1);
        }
        .upload-area svg { width: 48px; height: 48px; fill: #888; margin-bottom: 15px; }
        
        .controls {
            display: flex;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        .btn-primary {
            background: linear-gradient(135deg, #4facfe, #00f2fe);
            color: #000;
        }
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .btn-camera {
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: #fff;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        
        .results {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .result-box {
            text-align: center;
        }
        .result-box h3 {
            margin-bottom: 10px;
            color: #aaa;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .result-box img {
            max-width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .direction-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 10px 0;
        }
        .badge-photo { background: rgba(79, 172, 254, 0.2); color: #4facfe; }
        .badge-sketch { background: rgba(240, 147, 251, 0.2); color: #f093fb; }
        
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
        }
        .loading.active { display: block; }
        .spinner {
            width: 40px; height: 40px;
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid #4facfe;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        #camera-area { display: none; }
        #camera-area video {
            max-width: 100%;
            border-radius: 12px;
            margin-top: 10px;
        }
        
        .hidden { display: none !important; }
        
        select {
            padding: 10px 16px;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(255,255,255,0.1);
            color: #fff;
            font-size: 1em;
        }
        select option { background: #24243e; }
        
        @media (max-width: 600px) {
            .results { grid-template-columns: 1fr; }
            header h1 { font-size: 1.8em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎨 CycleGAN Face Translator</h1>
            <p>Upload a face photo or sketch — the model translates it automatically</p>
        </header>
        
        <div class="card">
            <div class="upload-area" id="upload-area" onclick="document.getElementById('file-input').click()">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                    <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
                </svg>
                <p style="font-size: 1.1em; margin-bottom: 5px;">Drop an image here or click to upload</p>
                <p style="color: #666; font-size: 0.9em;">Supports JPG, PNG images</p>
            </div>
            <input type="file" id="file-input" accept="image/*" style="display: none;">
            
            <div class="controls">
                <select id="direction-select">
                    <option value="auto">Auto-Detect Direction</option>
                    <option value="photo_to_sketch">Photo → Sketch</option>
                    <option value="sketch_to_photo">Sketch → Photo</option>
                </select>
                <button class="btn btn-primary" id="translate-btn" disabled onclick="translateImage()">
                    Translate
                </button>
                <button class="btn btn-camera" onclick="toggleCamera()">
                    📷 Camera
                </button>
            </div>
            
            <div id="camera-area">
                <video id="camera-video" autoplay playsinline></video>
                <div class="controls">
                    <button class="btn btn-primary" onclick="captureFromCamera()">📸 Capture</button>
                    <button class="btn btn-secondary" onclick="toggleCamera()">Close Camera</button>
                </div>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Translating image...</p>
        </div>
        
        <div class="card hidden" id="results-card">
            <div style="text-align: center; margin-bottom: 15px;">
                <span class="direction-badge" id="direction-badge"></span>
            </div>
            <div class="results">
                <div class="result-box">
                    <h3>Input</h3>
                    <img id="input-img" alt="Input">
                </div>
                <div class="result-box">
                    <h3>Output</h3>
                    <img id="output-img" alt="Output">
                </div>
            </div>
        </div>
    </div>
    
    <canvas id="capture-canvas" style="display: none;"></canvas>
    
    <script>
        let selectedFile = null;
        let cameraStream = null;
        
        // File input handler
        document.getElementById('file-input').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                document.getElementById('translate-btn').disabled = false;
                document.querySelector('.upload-area p:first-of-type').textContent = selectedFile.name;
            }
        });
        
        // Drag and drop
        const uploadArea = document.getElementById('upload-area');
        uploadArea.addEventListener('dragover', function(e) { e.preventDefault(); this.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', function() { this.classList.remove('dragover'); });
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault(); this.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                selectedFile = e.dataTransfer.files[0];
                document.getElementById('file-input').files = e.dataTransfer.files;
                document.getElementById('translate-btn').disabled = false;
                document.querySelector('.upload-area p:first-of-type').textContent = selectedFile.name;
            }
        });
        
        // Translate
        function translateImage() {
            if (!selectedFile) return;
            
            document.getElementById('loading').classList.add('active');
            document.getElementById('results-card').classList.add('hidden');
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            const direction = document.getElementById('direction-select').value;
            if (direction !== 'auto') formData.append('direction', direction);
            
            fetch('/translate', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    document.getElementById('loading').classList.remove('active');
                    if (data.success) {
                        document.getElementById('input-img').src = data.input_image;
                        document.getElementById('output-img').src = data.output_image;
                        const badge = document.getElementById('direction-badge');
                        badge.textContent = data.direction_label;
                        badge.className = 'direction-badge ' + 
                            (data.direction === 'photo_to_sketch' ? 'badge-photo' : 'badge-sketch');
                        document.getElementById('results-card').classList.remove('hidden');
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(err => {
                    document.getElementById('loading').classList.remove('active');
                    alert('Error: ' + err.message);
                });
        }
        
        // Camera
        function toggleCamera() {
            const area = document.getElementById('camera-area');
            if (area.style.display === 'none' || area.style.display === '') {
                navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
                    .then(stream => {
                        cameraStream = stream;
                        document.getElementById('camera-video').srcObject = stream;
                        area.style.display = 'block';
                    })
                    .catch(err => alert('Camera access denied: ' + err.message));
            } else {
                if (cameraStream) { cameraStream.getTracks().forEach(t => t.stop()); }
                area.style.display = 'none';
            }
        }
        
        function captureFromCamera() {
            const video = document.getElementById('camera-video');
            const canvas = document.getElementById('capture-canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            canvas.toBlob(function(blob) {
                selectedFile = new File([blob], 'camera_capture.png', { type: 'image/png' });
                document.getElementById('translate-btn').disabled = false;
                document.querySelector('.upload-area p:first-of-type').textContent = 'Camera capture ready';
                toggleCamera();
                translateImage();
            }, 'image/png');
        }
    </script>
</body>
</html>"""
    
    with open(os.path.join("templates", "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    print("[INFO] Template created.")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    create_templates()
    load_models()
    
    print("\n" + "=" * 50)
    print("  CycleGAN Web Interface")
    print("  Open: http://localhost:5000")
    print("=" * 50 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False)
