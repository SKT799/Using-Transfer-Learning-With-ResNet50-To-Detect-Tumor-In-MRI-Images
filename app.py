import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = 'uploads'
DEFAULT_IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEFAULT_IMAGE_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_brain_tumor_model.pth")
IMAGE_SIZE = 224
CLASS_NAMES = ['no', 'yes']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# PyTorch Preprocessing & Architecture
# ---------------------------------------------------------------------------
def crop_brain_contour(image_pil: Image.Image, add_pixels: int = 0) -> Image.Image:
    """Crop an MRI image to the bounding box of the largest brain contour."""
    image_np = np.array(image_pil)
    if image_np.ndim == 2:
        gray = image_np
    else:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_mask = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    binary_mask = cv2.erode(binary_mask, None, iterations=2)
    binary_mask = cv2.dilate(binary_mask, None, iterations=2)

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return image_pil

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    x = max(x - add_pixels, 0)
    y = max(y - add_pixels, 0)
    w = min(w + 2 * add_pixels, image_np.shape[1] - x)
    h = min(h + 2 * add_pixels, image_np.shape[0] - y)

    cropped = image_np[y: y + h, x: x + w]
    return Image.fromarray(cropped)

class BrainCropTransform:
    def __init__(self, add_pixels: int = 0):
        self.add_pixels = add_pixels

    def __call__(self, image_pil: Image.Image) -> Image.Image:
        return crop_brain_contour(image_pil, self.add_pixels)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

val_test_transforms = transforms.Compose([
    BrainCropTransform(add_pixels=2),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def build_brain_tumor_model(num_classes: int = 2):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 728),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(728, num_classes),
    )
    return model

# ---------------------------------------------------------------------------
# Load Model Setup
# ---------------------------------------------------------------------------
print("=" * 70)
print("Loading Fine-Tuned Brain Tumor Model")
print("=" * 70)

brain_tumor_model = build_brain_tumor_model(num_classes=2)
try:
    brain_tumor_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    brain_tumor_model = brain_tumor_model.to(device)
    brain_tumor_model.eval()
    print("[OK] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Error loading model weights from {MODEL_PATH}: {e}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_and_predict(image_path):
    """Run model inference on the provided image path."""
    try:
        raw_image = Image.open(image_path).convert("RGB")
        input_tensor = val_test_transforms(raw_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = brain_tumor_model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

        predicted_class = CLASS_NAMES[predicted_idx.item()]
        conf_pct = confidence.item() * 100
        
        # Details
        tumor_prob = probabilities[1].item() * 100
        no_tumor_prob = probabilities[0].item() * 100

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": conf_pct,
            "tumor_prob": tumor_prob,
            "no_tumor_prob": no_tumor_prob
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"success": False, "error": str(e)}

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/list-defaults', methods=['GET'])
def list_defaults():
    try:
        names = sorted([f for f in os.listdir(DEFAULT_IMAGE_FOLDER) if allowed_file(f)])
        return jsonify({'success': True, 'files': names})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load-defaults', methods=['POST'])
def load_defaults():
    try:
        data = request.get_json() or {}
        selected_file = data.get('selected_file')

        if not selected_file:
            return jsonify({'error': 'No image selected'}), 400

        fpath = os.path.join(DEFAULT_IMAGE_FOLDER, selected_file)
        if not os.path.isfile(fpath) or not allowed_file(selected_file):
            return jsonify({'error': 'Invalid file selected'}), 400

        print(f"\nEvaluating default image: {selected_file}")
        
        result = process_and_predict(fpath)
        if not result['success']:
            return jsonify({'error': result['error']}), 500

        return jsonify({
            'success': True,
            'source': 'default',
            'filename': selected_file,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'tumor_prob': result['tumor_prob'],
            'no_tumor_prob': result['no_tumor_prob'],
            'ground_truth': 'yes' if selected_file.lower().startswith('yes') else 'no'
        })

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        if file and allowed_file(file.filename):
            # Clear upload folder safely
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    pass

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            print(f"\nEvaluating uploaded image: {filename}")
            
            result = process_and_predict(filepath)
            
            if not result['success']:
                return jsonify({'error': result['error']}), 500

            return jsonify({
                'success': True,
                'source': 'uploaded',
                'filename': filename,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'tumor_prob': result['tumor_prob'],
                'no_tumor_prob': result['no_tumor_prob']
            })
            
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/serve-image/<source>/<filename>')
def serve_image(source, filename):
    safe_name = secure_filename(filename)
    if source == 'default':
        folder = DEFAULT_IMAGE_FOLDER
    elif source == 'uploaded':
        folder = os.path.abspath(UPLOAD_FOLDER)
    else:
        return 'Invalid source', 404
        
    return send_from_directory(folder, safe_name)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("Starting Flask Web Server...")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
