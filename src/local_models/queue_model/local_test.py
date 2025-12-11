# local_test.py
"""
Local test file - Manages database fetching, roi.json caching, and inference
Same invocation style as AWS version
"""

import os
import cv2
import base64
import json
import logging
import time
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Import modules
from db_manager import get_camera_config, check_for_updates, sync_all_cameras, load_roi_cache
from inference import model_fn, input_fn, predict_fn, output_fn

# Configuration
VIDEO_PATH = r"C:\Users\uct\Desktop\Q_Code\Vid1.mp4"  # Change this to your video path
CAMERA_ID = 1
USER_ID = 2
ORG_ID = 2
CHECK_UPDATE_INTERVAL = 120  # Check for DB updates every 30 seconds


def frame_to_b64(frame: np.ndarray) -> str:
    """Convert frame to base64 string"""
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode frame")
    return base64.b64encode(buf).decode("utf-8")


def b64_to_frame(b64_str: str) -> np.ndarray:
    """Convert base64 string to frame"""
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main():
    """Main test function"""
    
    print("=" * 70)
    print("Queue Monitoring System - Local Test")
    print("=" * 70)
    
    # Step 1: Load YOLO model
    print("\n[1/4] Loading YOLO model...")
    try:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model = model_fn(model_dir)
        print("✓ YOLO model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Step 2: Fetch camera config from database
    print(f"\n[2/4] Fetching camera {CAMERA_ID} configuration from database...")
    try:
        camera_config = get_camera_config(CAMERA_ID, force_refresh=False)
        if not camera_config:
            print(f"✗ Camera {CAMERA_ID} not found in database")
            print("\nTip: Check if camera exists in database or run sync:")
            print("  python -c 'from db_manager import sync_all_cameras; sync_all_cameras()'")
            return
        
        print(f"✓ Camera config loaded:")
        print(f"  - Region: {camera_config['region_name']}")
        print(f"  - Queues: {camera_config['number_of_queues']}")
        print(f"  - ROI coordinates cached in roi.json")
        
    except Exception as e:
        print(f"✗ Failed to fetch config: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Open video
    print(f"\n[3/4] Opening video: {VIDEO_PATH}")
    if not os.path.exists(VIDEO_PATH):
        print(f"✗ Video file not found: {VIDEO_PATH}")
        print("\nPlease set VIDEO_PATH in local_test.py to your video file")
        return
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"✗ Cannot open video: {VIDEO_PATH}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Video opened successfully")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps}")
    
    # Step 4: Process video frames
    print(f"\n[4/4] Processing video frames...")
    print("\nControls:")
    print("  Q - Quit")
    print("  P - Pause/Resume")
    print("  S - Save screenshot")
    print("  U - Force update from database")
    print()
    
    frame_count = 0
    last_update_check = time.time()
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✓ End of video reached")
                break
            
            frame_count += 1
            
            # Check for database updates periodically
            current_time = time.time()
            if current_time - last_update_check > CHECK_UPDATE_INTERVAL:
                logger.info("Checking for database updates...")
                if check_for_updates(CAMERA_ID):
                    logger.info("Database updates detected! Reloading config...")
                    camera_config = get_camera_config(CAMERA_ID, force_refresh=True)
                    # Note: In production, you'd recreate monitoring system here
                    logger.info("Config reloaded from database and cached to roi.json")
                last_update_check = current_time
            
            # Prepare payload (same as AWS SageMaker invocation)
            payload = {
                "camid": CAMERA_ID,
                "userid": USER_ID,
                "org_id": ORG_ID,
                "image": frame_to_b64(frame),
                "camera_config": camera_config,
                "return_annotated": True
            }
            
            # Invoke inference pipeline (same as AWS)
            try:
                # Step 1: Parse input
                input_data = input_fn(json.dumps(payload), "application/json")
                
                # Step 2: Run prediction
                result = predict_fn(input_data, model)
                
                # Step 3: Format output
                output_json = output_fn(result, "application/json")
                
                # Parse result
                result = json.loads(output_json)
                
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                continue
            
            # Display result
            if "Annotated_Frame" in result and result["Annotated_Frame"]:
                display_frame = b64_to_frame(result["Annotated_Frame"])
            else:
                display_frame = frame
            
            # Add info overlay
            '''y_offset = 30
            cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 35
            
            cv2.putText(display_frame, f"People: {result.get('Total_people_detected', 0)}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 35
            
            # Queue info
            for i, queue_name in enumerate(result.get('Queue_Name', [])):
                queue_len = result.get('Queue_Length', [0])[i]
                queue_status = result.get('Status', ['OK'])[i]
                status_color = (0, 255, 0) if queue_status == "OK" else (0, 0, 255)
                
                cv2.putText(display_frame, f"{queue_name}: {queue_len} [{queue_status}]",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                y_offset += 30
            
            '''
            cv2.imshow("Queue Monitoring - Local Test", display_frame)

        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n✓ Quit requested")
            break
        elif key == ord('p'):
            paused = not paused
            status = "Paused" if paused else "Resumed"
            print(f"  {status}")
        elif key == ord('s'):
            screenshot_name = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(screenshot_name, display_frame)
            print(f"  Screenshot saved: {screenshot_name}")
        elif key == ord('u'):
            print("  Force updating from database...")
            camera_config = get_camera_config(CAMERA_ID, force_refresh=True)
            print("  ✓ Config updated and cached to roi.json")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print(f"Processing complete: {frame_count} frames processed")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    # Allow video path as command line argument
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
