from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import model_from_json
import tensorflow.keras as keras
import numpy as np
import cv2
import base64
import logging
from typing import Tuple, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_JSON_PATH = "model.json"
MODEL_WEIGHTS_PATH = "model/Brain_Tumor_Model.h5"
IMAGE_SIZE = (224, 224)
TUMOR_THRESHOLD = 50.0  # Percentage threshold for tumor detection
HOST = "127.0.0.1"
PORT = 5001

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model variable
model = None


def load_model() -> keras.models.Model:
    """
    Load the pre-trained brain tumor detection model.
    
    Returns:
        keras.models.Model: Loaded Keras model
        
    Raises:
        FileNotFoundError: If model files are not found
        Exception: If model loading fails
    """
    try:
        logger.info("Loading model architecture from %s", MODEL_JSON_PATH)
        
        if not Path(MODEL_JSON_PATH).exists():
            raise FileNotFoundError(f"Model JSON file not found: {MODEL_JSON_PATH}")
        
        if not Path(MODEL_WEIGHTS_PATH).exists():
            raise FileNotFoundError(f"Model weights file not found: {MODEL_WEIGHTS_PATH}")
        
        with keras.utils.custom_object_scope({'Functional': keras.models.Model}):
            with open(MODEL_JSON_PATH, "r") as json_file:
                loaded_model_json = json_file.read()
            
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(MODEL_WEIGHTS_PATH)
        
        logger.info("Model loaded successfully")
        return loaded_model
        
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        raise


# Load model at startup
try:
    model = load_model()
except Exception as e:
    logger.critical("Failed to initialize model. Application may not function correctly.")
    raise


def decode_base64_image(b64str: str) -> np.ndarray:
    """
    Decode a base64-encoded image string to a numpy array.
    
    Args:
        b64str: Base64-encoded image string (with or without data URI prefix)
        
    Returns:
        np.ndarray: Decoded image as numpy array in BGR format
        
    Raises:
        ValueError: If the base64 string is invalid or cannot be decoded
    """
    try:
        # Remove data URI prefix if present
        if "," in b64str:
            encoded_data = b64str.split(",")[1]
        else:
            encoded_data = b64str
        
        # Decode base64 to bytes
        img_bytes = base64.b64decode(encoded_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image from base64 string")
        
        return img
        
    except Exception as e:
        logger.error("Error decoding base64 image: %s", str(e))
        raise ValueError(f"Invalid base64 image data: {str(e)}")


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for model prediction.
    
    Args:
        img: Input image as numpy array
        
    Returns:
        np.ndarray: Preprocessed image ready for model input
    """
    # Resize to model input size
    img_resized = cv2.resize(img, IMAGE_SIZE)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_resized, axis=0)
    
    return img_batch


def predict_tumor(img: np.ndarray) -> Tuple[float, float, str]:
    """
    Predict tumor presence from an image.
    
    Args:
        img: Preprocessed image array
        
    Returns:
        Tuple containing:
            - tumor_probability: Probability of tumor presence (0-100)
            - normal_probability: Probability of no tumor (0-100)
            - result: Classification result string
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Get prediction
    prediction = model.predict(img, verbose=0)[0]
    
    # Model outputs [normal_probability, tumor_probability]
    normal_prob = float(prediction[0] * 100)
    tumor_prob = float(prediction[1] * 100)
    
    # Determine result
    result = "TUMOR DETECTED" if tumor_prob >= TUMOR_THRESHOLD else "NO TUMOR"
    
    logger.info("Prediction - Tumor: %.2f%%, Normal: %.2f%%, Result: %s", 
                tumor_prob, normal_prob, result)
    
    return tumor_prob, normal_prob, result


@app.route("/")
def index():
    """
    Serve the main application page.
    
    Returns:
        HTML template for the web interface
    """
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify API status.
    
    Returns:
        JSON response with API status and model information
    """
    return jsonify({
        "status": "healthy",
        "service": "Brain Tumor Detection API",
        "version": "1.0.0",
        "model_loaded": model is not None
    })
@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    """
    Predict tumor from base64-encoded image.
    
    Expected JSON payload:
        {
            "image": ["data:image/jpeg;base64,..."]
        }
    
    Returns:
        JSON response with prediction results
    """
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        if "image" not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        if not isinstance(data["image"], list) or len(data["image"]) == 0:
            return jsonify({"error": "Image must be a non-empty array"}), 400
        
        # Decode and preprocess image
        b64_data = data["image"][0]
        img = decode_base64_image(b64_data)
        img_preprocessed = preprocess_image(img)
        
        # Predict
        tumor_prob, normal_prob, result = predict_tumor(img_preprocessed)
        
        # Return results
        return jsonify({
            "success": True,
            "result": result,
            "tumor_probability": round(tumor_prob, 2),
            "normal_probability": round(normal_prob, 2),
            "confidence": round(max(tumor_prob, normal_prob), 2)
        })
        
    except ValueError as e:
        logger.warning("Validation error: %s", str(e))
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error("Prediction error: %s", str(e), exc_info=True)
        return jsonify({"success": False, "error": "Internal server error"}), 500

@app.route("/predict_string", methods=["GET"])
def predict_string():
    """
    Predict tumor from base64 string passed as query parameter.
    
    Query Parameters:
        image: Base64-encoded image string
    
    Returns:
        JSON response with prediction results
    """
    try:
        # Get and validate query parameter
        b64_data = request.args.get("image")
        
        if not b64_data:
            return jsonify({"error": "Base64 image string not provided"}), 400
        
        # Decode and preprocess image
        img = decode_base64_image(b64_data)
        img_preprocessed = preprocess_image(img)
        
        # Predict
        tumor_prob, normal_prob, result = predict_tumor(img_preprocessed)
        
        # Return results
        return jsonify({
            "success": True,
            "result": result,
            "tumor_probability": round(tumor_prob, 2),
            "normal_probability": round(normal_prob, 2),
            "confidence": round(max(tumor_prob, normal_prob), 2)
        })
        
    except ValueError as e:
        logger.warning("Validation error: %s", str(e))
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error("Prediction error: %s", str(e), exc_info=True)
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict tumor from uploaded image file.
    
    Form Data:
        image: Image file (multipart/form-data)
    
    Returns:
        JSON response with prediction results
    """
    try:
        # Validate file upload
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["image"]
        
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            }), 400
        
        # Read and decode image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Failed to decode image file"}), 400
        
        # Preprocess and predict
        img_preprocessed = preprocess_image(img)
        tumor_prob, normal_prob, result = predict_tumor(img_preprocessed)
        
        # Return results
        return jsonify({
            "success": True,
            "result": result,
            "tumor_probability": round(tumor_prob, 2),
            "normal_probability": round(normal_prob, 2),
            "confidence": round(max(tumor_prob, normal_prob), 2),
            "filename": file.filename
        })
        
    except ValueError as e:
        logger.warning("Validation error: %s", str(e))
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error("Prediction error: %s", str(e), exc_info=True)
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error("Internal server error: %s", str(error))
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting Brain Tumor Detection API")
    logger.info("Server running at http://%s:%d", HOST, PORT)
    logger.info("Available endpoints:")
    logger.info("  GET  /              - Web interface")
    logger.info("  GET  /health        - Health check")
    logger.info("  POST /predict       - File upload prediction")
    logger.info("  POST /predict_base64 - Base64 prediction")
    logger.info("  GET  /predict_string - Query string prediction")
    
    app.run(host=HOST, port=PORT, debug=True)
