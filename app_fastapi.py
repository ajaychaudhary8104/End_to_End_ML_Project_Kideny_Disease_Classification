from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import traceback
from typing import Optional
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    logger.info("FastAPI server starting...")
    logger.info(f"Model loaded: {model_app.model is not None}")
    yield
    # Shutdown
    logger.info("FastAPI server shutting down...")


# FastAPI app
app = FastAPI(
    title="Kidney Disease Classification API",
    description="CT Scan Image Analysis using Deep Learning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


class PredictionModel:
    """Model wrapper for predictions"""
    
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            model_path = os.path.join("artifacts", "training", "model.keras")
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                logger.info(f"Model loaded successfully from {model_path}")
            else:
                logger.warning(f"Warning: Model not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            traceback.print_exc()

    def predict(self, image_path: str) -> dict:
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
            predictions = self.model.predict(test_image, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])

            # Map class to label
            class_labels = {0: "Normal", 1: "Tumor"}
            prediction_label = class_labels.get(predicted_class, "Unknown")

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
            logger.error(f"Error during prediction: {str(e)}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }


# Initialize model
model_app = PredictionModel()


def cleanup_file(file_path: str):
    """Background task to cleanup uploaded file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file: {str(e)}")


# Health check response model
class HealthResponse:
    status: str
    model_loaded: bool


# Prediction response model
class PredictionResponse:
    success: bool
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    class_index: Optional[int] = None
    raw_predictions: Optional[dict] = None
    error: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the prediction UI"""
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Index page not found</h1>"


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model_app.model is not None,
        "service": "Kidney Disease Classification API"
    }


@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": "Kidney Disease Classification API",
        "version": "1.0.0",
        "description": "CT Scan Image Analysis using Deep Learning",
        "endpoints": {
            "GET /": "Serve UI",
            "GET /health": "Health check",
            "GET /api/info": "API information",
            "POST /predict": "Make prediction",
            "POST /train": "Start training"
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Predict kidney disease from CT scan image
    
    - **file**: Upload a CT scan image (PNG, JPG, JPEG)
    """
    file_path = None
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file selected"
            )

        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail="File type not allowed. Use PNG, JPG, or JPEG"
            )

        # Check file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds 10MB limit"
            )

        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        # Make prediction
        result = model_app.predict(file_path)

        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(cleanup_file, file_path)
        else:
            # Fallback: cleanup immediately
            cleanup_file(file_path)

        if result.get("success"):
            return result
        else:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Prediction failed")
            )

    except HTTPException:
        # Cleanup on error
        if file_path:
            cleanup_file(file_path)
        raise
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        traceback.print_exc()
        # Cleanup on error
        if file_path:
            cleanup_file(file_path)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/train")
async def train(background_tasks: BackgroundTasks = None):
    """
    Start model training
    
    This endpoint triggers the training pipeline by running main.py
    """
    try:
        def run_training():
            logger.info("Starting training...")
            exit_code = os.system("python main.py")
            if exit_code == 0:
                logger.info("Training completed successfully")
                # Reload model after training
                model_app.load_model()
            else:
                logger.error(f"Training failed with exit code: {exit_code}")

        if background_tasks:
            background_tasks.add_task(run_training)
            return {
                "success": True,
                "message": "Training started in background. Check logs for progress."
            }
        else:
            run_training()
            return {
                "success": True,
                "message": "Training completed successfully!"
            }

    except Exception as e:
        logger.error(f"Error in train endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/api/model-info")
async def model_info():
    """Get model information"""
    try:
        return {
            "model_loaded": model_app.model is not None,
            "model_path": "artifacts/training/model.keras",
            "input_shape": [224, 224, 3],
            "classes": ["Normal", "Tumor"],
            "framework": "TensorFlow/Keras"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "success": False,
        "error": "Internal server error"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
