"""
Flask API server for MNIST Digit Recognition.
Supports single and multi-digit recognition via contour segmentation.
"""

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
from scipy import ndimage
import os

app = Flask(__name__, static_folder="static")
CORS(app)

# ═══════════════════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "saved_model/mnist_advanced.keras"
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run main.py first.")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    return _model


# ═══════════════════════════════════════════════════════════════════════════════
#  Digit Segmentation
# ═══════════════════════════════════════════════════════════════════════════════

def segment_digits(img_array):
    """
    Segment a grayscale image into individual digit bounding boxes.

    Uses connected component labeling to find separate digit blobs,
    then sorts them left-to-right.

    Args:
        img_array: 2D numpy array (H, W), values in [0, 1], white-on-black.

    Returns:
        List of dicts with keys: 'image' (28x28 array), 'bbox' (x, y, w, h)
        sorted left-to-right. Returns empty list if no digits found.
    """
    # Threshold to binary
    binary = (img_array > 0.15).astype(np.uint8)

    if binary.sum() < 10:
        return []

    # Label connected components
    labeled, num_features = ndimage.label(binary)

    if num_features == 0:
        return []

    digits = []
    for i in range(1, num_features + 1):
        # Get bounding box for this component
        component_mask = (labeled == i)
        rows = np.where(component_mask.any(axis=1))[0]
        cols = np.where(component_mask.any(axis=0))[0]

        if len(rows) == 0 or len(cols) == 0:
            continue

        y_min, y_max = rows[0], rows[-1]
        x_min, x_max = cols[0], cols[-1]
        h = y_max - y_min + 1
        w = x_max - x_min + 1

        # Filter out noise — skip tiny components
        if h < 8 or w < 5 or (h * w) < 50:
            continue

        # Extract digit region
        digit_crop = img_array[y_min:y_max + 1, x_min:x_max + 1]

        # Pad to square with border
        max_dim = max(h, w)
        pad = int(max_dim * 0.3)  # 30% padding
        padded_size = max_dim + 2 * pad

        padded = np.zeros((padded_size, padded_size), dtype=np.float32)
        y_offset = (padded_size - h) // 2
        x_offset = (padded_size - w) // 2
        padded[y_offset:y_offset + h, x_offset:x_offset + w] = digit_crop

        # Resize to 28x28 using PIL
        pil_img = Image.fromarray((padded * 255).astype(np.uint8), mode='L')
        pil_img = pil_img.resize((28, 28), Image.LANCZOS)
        digit_28 = np.array(pil_img).astype(np.float32) / 255.0

        digits.append({
            'image': digit_28,
            'bbox': {'x': int(x_min), 'y': int(y_min), 'w': int(w), 'h': int(h)},
            'center_x': (x_min + x_max) / 2,
        })

    # Sort left-to-right
    digits.sort(key=lambda d: d['center_x'])

    return digits


def preprocess_single_digit(img_array):
    """Preprocess a single full-canvas digit (original behavior)."""
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array
    return img_array


def predict_digit_image(model, digit_28):
    """Run prediction on a single 28x28 digit image."""
    img_input = digit_28.reshape(1, 28, 28, 1)
    predictions = model.predict(img_input, verbose=0)[0]
    pred_label = int(np.argmax(predictions))
    confidence = float(predictions[pred_label]) * 100
    return pred_label, confidence, predictions


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """Serve a minimal favicon to prevent 404."""
    pixel = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAB"
        "Nl7BcQAAAABJRU5ErkJggg=="
    )
    return Response(pixel, mimetype="image/png")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept base64 canvas image, segment digits, return predictions."""
    try:
        data = request.get_json()
        image_data = data.get("image", "")

        # Decode base64 image
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes)).convert("L")
        img_array = np.array(img).astype(np.float32) / 255.0

        # Invert if background is light
        if img_array.mean() > 0.5:
            img_array = 1.0 - img_array

        # Check if blank
        if img_array.max() < 0.05:
            return jsonify({"error": "blank", "message": "Please draw a digit first."})

        model = get_model()

        # Try multi-digit segmentation first
        digit_segments = segment_digits(img_array)

        if len(digit_segments) == 0:
            # Fallback: treat entire image as single digit
            resized = np.array(
                Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
                .resize((28, 28), Image.LANCZOS)
            ).astype(np.float32) / 255.0
            digit_segments = [{'image': resized, 'bbox': {'x': 0, 'y': 0, 'w': img.width, 'h': img.height}}]

        # Predict each digit
        results = []
        for seg in digit_segments:
            pred_label, confidence, preds = predict_digit_image(model, seg['image'])
            results.append({
                'digit': pred_label,
                'confidence': round(confidence, 2),
                'confidences': {str(i): round(float(preds[i]) * 100, 2) for i in range(10)},
                'bbox': seg['bbox'],
            })

        # Build combined number
        combined_number = ''.join(str(r['digit']) for r in results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)

        # For single digit, also return legacy format for backward compat
        if len(results) == 1:
            r = results[0]
            top3_indices = np.argsort([r['confidences'][str(i)] for i in range(10)])[::-1][:3]
            return jsonify({
                "prediction": r['digit'],
                "confidence": r['confidence'],
                "confidences": r['confidences'],
                "top3": [{"digit": int(i), "confidence": r['confidences'][str(i)]} for i in top3_indices],
                "combined_number": combined_number,
                "digit_count": 1,
                "digits": results,
            })

        return jsonify({
            "combined_number": combined_number,
            "avg_confidence": round(avg_confidence, 2),
            "digit_count": len(results),
            "digits": results,
            # Legacy compat — use first digit
            "prediction": results[0]['digit'],
            "confidence": results[0]['confidence'],
            "confidences": results[0]['confidences'],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🔄 Loading model...")
    get_model()
    print("🚀 Starting server on http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)
