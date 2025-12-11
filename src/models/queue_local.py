import cv2
import json
import base64
import numpy as np
import logging
import os
from dotenv import load_dotenv

import sys, os

# Add <project_root>/src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.local_models.queue_model.inference import input_fn,model_fn, predict_fn,output_fn

# ---------- Setup ----------
model_dir = r"..\local_models\queue_model\model.pt"  
model_info = model_fn(model_dir)
#------------------------------------------------------------------------------- PPE Detection ------------------------------------------------------------------------------

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger("detection")
logger.setLevel(logging.INFO)




# -------------------------------------------------------------------------------
# Queue monitoring
# ------------------------------------------------------------------------------


def queue_monitering(frame, camera_id, user_id, org_id,camera_config):
    """Send a frame to SageMaker endpoint and return (result, error_message, annotated_frame) safely."""
    try:

         # Prepare payload (same as AWS SageMaker invocation)
        payload = {
                "camid": camera_id,
                "userid": user_id,
                "org_id": org_id,
                "image": frame,
                "camera_config": camera_config,
                "return_annotated": True
            }

        
        try:

            # Step 1: Parse input
            input_data = input_fn(json.dumps(payload), "application/json") 
            # Step 2: Run prediction
            output = predict_fn(input_data, model_info)
            output=output_fn(output, "application/json")
            result = json.loads(output)
            
            return result,None
        
        except (json.JSONDecodeError, AttributeError) as e:
            msg = f"Invalid JSON response from SageMaker: {e}"
            logger.error(msg)
            return None, msg

    except Exception as e:
        msg = f"Unexpected error in people_counting: {e}"
        logger.exception(msg)
        return None, msg

   