# local_test.py
"""
Local test file - Manages database fetching, roi.json caching, and inference
Same invocation style as AWS version
Enhanced with customer visit tracking display + Excel logging (batch writing)
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
from src.local_models.queue_model.db_manager import get_camera_config, check_for_updates, sync_all_cameras, \
    load_roi_cache
from src.local_models.queue_model.inference import model_fn, input_fn, predict_fn, output_fn
from excel_logger import ExcelLogger, save_results  # ← Updated import

# Configuration
VIDEO_PATH = r"E:\UTC project\CCTV_Project\Video\Vid.mp4"  # Change this to your video path
CAMERA_ID = 3135
USER_ID = 4
ORG_ID = 4
CHECK_UPDATE_INTERVAL = 120  # Check for DB updates every 120 seconds


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


def convert_normalized_to_absolute_coords(camera_config: Dict[str, Any], frame_width: int, frame_height: int) -> Dict[
    str, Any]:
    """
    Convert normalized coordinates (0-1) to absolute pixel coordinates

    Args:
        camera_config: Camera configuration with normalized coordinates
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        Updated camera config with absolute coordinates
    """
    config_copy = camera_config.copy()

    if 'queues_coordinates' in config_copy:
        for queue in config_copy['queues_coordinates']:
            if 'rect' in queue and isinstance(queue['rect'], dict):
                rect = queue['rect']

                # Check if coordinates are normalized (between 0 and 1)
                if all(0 <= rect.get(k, 2) <= 1 for k in ['x', 'y', 'w', 'h']):
                    # Convert to absolute coordinates
                    queue['rect'] = {
                        'x': int(rect['x'] * frame_width),
                        'y': int(rect['y'] * frame_height),
                        'w': int(rect['w'] * frame_width),
                        'h': int(rect['h'] * frame_height)
                    }
                    logger.debug(f"Converted Queue {queue['queue_id']} coordinates to absolute: {queue['rect']}")

    return config_copy


def main():
    """Main test function"""

    print("=" * 70)
    print("Queue Monitoring System - Local Test")
    print("=" * 70)

    # Initialize Excel logger (in-memory, no disk writes yet)
    excel_logger = ExcelLogger()
    print("✓ Excel logger initialized (batch mode - results will be saved at the end)")

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
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✓ Video opened successfully")
    print(f"  - Resolution: {frame_width}x{frame_height}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps}")

    # IMPORTANT: Convert normalized coordinates to absolute coordinates
    print(f"\n[3.5/4] Converting normalized coordinates to absolute...")
    camera_config = convert_normalized_to_absolute_coords(camera_config, frame_width, frame_height)
    print("✓ Coordinates converted")

    # Step 4: Process video frames
    print(f"\n[4/4] Processing video frames...")
    print("\nControls:")
    print("  Q - Quit")
    print("  P - Pause/Resume")
    print("  S - Save screenshot")
    print("  U - Force update from database")
    print("  D - Display stats")
    print()

    frame_count = 0
    last_update_check = time.time()
    paused = False
    last_result = None
    start_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✓ End of video reached")
                break

            frame_count += 1

            # Progress indicator
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processing = frame_count / elapsed if elapsed > 0 else 0
                print(f"  Processed {frame_count}/{total_frames} frames ({fps_processing:.1f} fps)")

            # Check for database updates periodically
            current_time = time.time()
            if current_time - last_update_check > CHECK_UPDATE_INTERVAL:
                logger.info("Checking for database updates...")
                if check_for_updates(CAMERA_ID):
                    logger.info("Database updates detected! Reloading config...")
                    new_config = get_camera_config(CAMERA_ID, force_refresh=True)
                    camera_config = convert_normalized_to_absolute_coords(new_config, frame_width, frame_height)
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
                last_result = result

                # ── Add to Excel DataFrame (in-memory, no disk write) ──────────────────
                try:
                    # Remove annotated frame to save memory
                    result_for_log = {k: v for k, v in result.items() if k != "Annotated_Frame"}
                    excel_logger.add_result(result_for_log)
                except Exception as excel_err:
                    logger.warning(f"Excel logging failed for frame {frame_count}: {excel_err}")
                # ─────────────────────────────────────────────────────────────────────

            except Exception as e:
                logger.error(f"Inference failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Display result
            if "Annotated_Frame" in result and result["Annotated_Frame"]:
                display_frame = b64_to_frame(result["Annotated_Frame"])
            else:
                display_frame = frame

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
            new_config = get_camera_config(CAMERA_ID, force_refresh=True)
            camera_config = convert_normalized_to_absolute_coords(new_config, frame_width, frame_height)
            print("  ✓ Config updated and cached to roi.json")
        elif key == ord('d'):
            if last_result:
                print("\n" + "=" * 70)
                print("CURRENT STATISTICS")
                print("=" * 70)
                print(f"Total Customers Visited: {last_result.get('Total_Customer_Visited', 0)}")
                print(f"Current People Detected: {last_result.get('Total_people_detected', 0)}")

                queue_names = last_result.get('Queue_Name', [])
                queue_customers = last_result.get('Queue_Customer_Visited', [])
                queue_lengths = last_result.get('Queue_Length', [])

                for i, queue_name in enumerate(queue_names):
                    customers = queue_customers[i] if i < len(queue_customers) else 0
                    length = queue_lengths[i] if i < len(queue_lengths) else 0
                    print(f"\n{queue_name}:")
                    print(f"  - Total Visited: {customers}")
                    print(f"  - Current Length: {length}")

                tracker_stats = last_result.get('Tracker_Stats', {})
                if tracker_stats:
                    print(f"\nTracker Stats:")
                    print(f"  - Active Tracks: {tracker_stats.get('active_tracks', 0)}")
                    print(f"  - Unique Persons: {tracker_stats.get('unique_persons', 0)}")
                    print(f"  - Total IDs Created: {tracker_stats.get('total_tracks_created', 0)}")

                # Show memory usage info
                memory_rows = len(excel_logger.results_df)
                print(f"\nExcel Buffer:")
                print(f"  - Rows in memory: {memory_rows}")
                print(f"  - (Will be saved on exit)")

                print("=" * 70 + "\n")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    # Final statistics and save Excel file
    print("\n" + "=" * 70)
    print("SAVING RESULTS TO EXCEL")
    print("=" * 70)

    # Save all accumulated results to Excel (single write operation)
    excel_logger.save_to_excel()

    # Final statistics
    if last_result:
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        print(f"Frames Processed: {frame_count}")
        print(f"Total Unique Customers Visited: {last_result.get('Total_Customer_Visited', 0)}")

        queue_names = last_result.get('Queue_Name', [])
        queue_customers = last_result.get('Queue_Customer_Visited', [])

        print("\nPer-Queue Statistics:")
        for i, queue_name in enumerate(queue_names):
            customers = queue_customers[i] if i < len(queue_customers) else 0
            print(f"  {queue_name}: {customers} unique customers")

        tracker_stats = last_result.get('Tracker_Stats', {})
        if tracker_stats:
            print(f"\nTracker Summary:")
            print(f"  - Unique Persons Detected: {tracker_stats.get('unique_persons', 0)}")
            print(f"  - Total Track IDs Generated: {tracker_stats.get('total_tracks_created', 0)}")

        processing_time = time.time() - start_time
        avg_fps = frame_count / processing_time if processing_time > 0 else 0
        print(f"\nPerformance:")
        print(f"  - Total processing time: {processing_time:.2f} seconds")
        print(f"  - Average processing FPS: {avg_fps:.2f}")
        print(f"  - Excel write operations: 1 (instead of {frame_count})")

        print("=" * 70)
    else:
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
        # Try to save results on interrupt
        try:
            from excel_logger import save_results

            save_results()
            print("✓ Results saved before exit")
        except:
            pass
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()