# inference.py
"""
Inference module - same interface as AWS SageMaker version
Load YOLO model, process frames with queue monitoring
"""

import os
import json
import base64
import logging
from typing import Dict, Any
import numpy as np
import cv2
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available - install: pip install ultralytics")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import queue monitoring
from src.local_models.queue_model.queue_monitoring import QueueMonitoringSystem
# from queue_monitoring import QueueMonitoringSystem

# Global state
yolo_model = None
monitoring_systems: Dict[int, QueueMonitoringSystem] = {}


def model_fn(model_dir: str):
    """
    Load YOLO model - same as AWS SageMaker version
    
    Args:
        model_dir: Directory containing model file or path to model
    
    Returns:
        Loaded YOLO model
    """
    global yolo_model
    
    if not YOLO_AVAILABLE:
        raise RuntimeError("YOLO not available. Install: pip install ultralytics")
    
    # Look for model file in directory
    model_path = None
    
    if os.path.isfile(model_dir):
        # model_dir is actually a file path
        model_path = model_dir
    elif os.path.isdir(model_dir):
        # Look for .pt files in directory
        for file in os.listdir(model_dir):
            if file.endswith('.pt'):
                model_path = os.path.join(model_dir, file)
                break
    
    # Default to yolov8n.pt if not found
    if not model_path or not os.path.exists(model_path):
        model_path = "yolov8n.pt"
        logger.info("Using default model: yolov8n.pt")
    
    try:
        yolo_model = YOLO(model_path)
        logger.info(f"YOLO model loaded: {model_path}")
        return yolo_model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise


def input_fn(request_body: str, content_type: str = "application/json"):
    """
    Parse input request - same as AWS SageMaker version
    
    Expected JSON format:
    {
        "camid": 1,
        "userid": 2,
        "org_id": 2,
        "image": "base64_encoded_image",
        "camera_config": {...},  # Optional, will be fetched if not provided
        "return_annotated": true
    }
    
    Args:
        request_body: JSON string
        content_type: Content type
    
    Returns:
        Parsed input data
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    
    try:
        data = json.loads(request_body)
        
        # Decode image
        image_b64 = data.get("image", "")
        if not image_b64:
            raise ValueError("No image provided")
        
        image_bytes = base64.b64decode(image_b64)
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image")
        
        data['decoded_frame'] = frame
        
        return data
        
    except Exception as e:
        logger.error(f"Error parsing input: {e}")
        raise


def predict_fn(input_data: Dict[str, Any], model) -> Dict[str, Any]:
    """
    Run inference - same as AWS SageMaker version
    
    Args:
        input_data: Parsed input data from input_fn
        model: Loaded YOLO model
    
    Returns:
        Prediction result with queue monitoring data
    """
    global monitoring_systems
    
    try:
        camera_id = input_data.get("camid")
        frame = input_data.get("decoded_frame")
        camera_config = input_data.get("camera_config")
        return_annotated = input_data.get("return_annotated", True)
        
        if camera_id is None:
            raise ValueError("Camera ID not provided")
        
        if frame is None:
            raise ValueError("Frame not decoded")
        
        if camera_config is None:
            raise ValueError("Camera config not provided")
        
        # Get or create monitoring system
        if camera_id not in monitoring_systems:
            monitoring_systems[camera_id] = QueueMonitoringSystem(model, camera_config)
            logger.info(f"Created monitoring system for camera {camera_id}")
        
        # Process frame
        result = monitoring_systems[camera_id].process_frame(frame, return_annotated)
        
        # Add metadata
        result['camid'] = camera_id
        result['userid'] = input_data.get('userid', 0)
        result['org_id'] = input_data.get('org_id', 0)
        result['Processing_Status'] = 1
        
        # Encode annotated frame if present
        if result.get('Annotated_Frame') is not None:
            _, buffer = cv2.imencode('.jpg', result['Annotated_Frame'])
            result['Annotated_Frame'] = base64.b64encode(buffer).decode('utf-8')
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            "camid": input_data.get("camid", 0),
            "userid": input_data.get("userid", 0),
            "org_id": input_data.get("org_id", 0),
            "Processing_Status": 0,
            "Error": str(e)
        }


def output_fn(prediction: Dict[str, Any], accept: str = "application/json") -> str:
    """
    Format output - same as AWS SageMaker version
    
    Args:
        prediction: Prediction result
        accept: Accept content type
    
    Returns:
        JSON string
    """
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}")
    
    try:
        return json.dumps(prediction, separators=(',', ':'))
    except Exception as e:
        logger.error(f"JSON serialization failed: {e}")
        return json.dumps({"Error": str(e), "Processing_Status": 0})


if __name__ == "__main__":
    # Test inference module
    logger.info("Testing inference module...")
    
    try:
        # Load model
        model = model_fn("yolov8n.pt")
        logger.info("✓ Model loaded successfully")
        
        logger.info("Inference module ready!")
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
