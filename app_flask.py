from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.pipeline.prediction import PredictionPipeline
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import traceback

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class ClientApp:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            model_path = os.path.join("artifacts", "training", "model.keras")
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                print(f"Model loaded successfully from {model_path}")
            else:
                print(f"Warning: Model not found at {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()

    def predict(self, image_path):
        """Make prediction on image"""
        try:
            if self.model is None:
                return {
                    "success": False,
                    "error": "Model not loaded"
                }

            # Load and preprocess image
            test_image = image.load_img(image_path, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image / 255.0  # Normalize

            # Make prediction
            predictions = self.model.predict(test_image)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            # Map class to label
            class_labels = {0: 'Normal', 1: 'Tumor'}
            prediction_label = class_labels.get(predicted_class, 'Unknown')

            return {
                "success": True,
                "prediction": prediction_label.lower(),
                "confidence": confidence,
                "class_index": int(predicted_class),
                "raw_predictions": {
                    "Normal": float(predictions[0][0]),
                    "Tumor": float(predictions[0][1])
                }
            }
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    """Serve the prediction UI"""
    return render_template('index.html')


@app.route("/health", methods=['GET'])
@cross_origin()
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": clApp.model is not None
    })


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    """Train the model"""
    try:
        os.system("python main.py")
        return jsonify({
            "success": True,
            "message": "Training completed successfully!"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    """Predict kidney disease from CT scan image"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400

        file = request.files['file']

        # Check if file is empty
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400

        # Validate file
        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "File type not allowed. Use PNG, JPG, or JPEG"
            }), 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        result = clApp.predict(filepath)

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        if result.get("success"):
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        "success": False,
        "error": "File size exceeds 10MB limit"
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "success": False,
        "error": "Endpoint not found"
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(debug=True, host='0.0.0.0', port=5000)

    #app.run(host='0.0.0.0', port=8080) #for AWS


