import cv2
import json
import time
import asyncio
import logging
import sys
import os
import base64
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue
# import threading
import multiprocessing

from src.store_s3.queue_store import upload_to_s3
from src.database.queue_query import insert_data

# # Add <project_root>/src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.local_models.queue_model.db_manager import get_camera_config, check_for_updates


from src.models.queue_local import queue_monitering


logger = logging.getLogger("people_counting")
logger.setLevel(logging.INFO)
CHECK_UPDATE_INTERVAL = 120


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



def run_storage_worker(q, client_id):
    """Thread that stores frames sequentially (no duplicates & no missing frames)."""
    while True:
        item = q.get()
        if item is None:
            break

        frame_id, annotated_frame, detections = item

        try:
            s3_url = upload_to_s3(annotated_frame, frame_id)
            insert_data(detections, s3_url)
            logger.info(f"[{client_id}] Stored frame {frame_id}")
        except Exception as e:
            logger.error(f"[{client_id}] Error storing frame {frame_id}: {e}")

        q.task_done()



def run_queuemonitoring_detection(
    client_id: str,
    video_url: str,
    camera_id: int,
    user_id: int,
    org_id: int,
    sessions: dict,
    loop: asyncio.AbstractEventLoop,
    storage_executor: ThreadPoolExecutor
):
    """
    Runs People Counting detection in a separate thread.
    Sends WebSocket messages safely and stores frames to S3/DB in background threads to avoid blocking inference.
    """
    camera_config = get_camera_config(camera_id, force_refresh=False)
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        logger.error(f"[{client_id}] Unable to open video stream: {video_url}")
        return

    frame_num = 0
    last_update_check = time.time()

    # Thread-safe FIFO queue for storage tasks
    store_queue = Queue(maxsize=1000)

    # Start the storage worker thread
    storage_process = multiprocessing.Process(
    target=run_storage_worker,
    args=(store_queue, client_id),
    daemon=True
    )
    storage_process.start()

    while cap.isOpened() and sessions.get(client_id, {}).get("streaming", False):
        ret, frame = cap.read()
        if not ret:
            continue

        frame_num += 1

        # Check for database updates periodically  
        # Need to be check for why he (abhishek) put in while loop ,need to change this
        
        current_time = time.time()
        if current_time - last_update_check > CHECK_UPDATE_INTERVAL:
            logger.info("Checking for database updates...")
            if check_for_updates(camera_id):
                logger.info("Database updates detected! Reloading config...")
                camera_config = get_camera_config(camera_id, force_refresh=True)
                # Note: In production, you'd recreate monitoring system here
                logger.info("Config reloaded from database and cached to roi.json")
            last_update_check = current_time

        # if frame_num%2==0:
        #     continue

    
        frame_base64 = frame_to_b64(frame)

        try:

            # ---------------- People Counting Inference ----------------
            detections, error = queue_monitering(frame_base64, camera_id, user_id, org_id,camera_config)

            

            ws = sessions.get(client_id, {}).get("ws")
            

            if detections:
                payload = {"detections": detections}

                # ---------------- WebSocket Send ----------------
                if ws:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            ws.send_text(json.dumps(payload)),
                            loop
                        )
                        logger.info(f"[{client_id}] Frame {frame_num}: Detections sent to client")
                    except Exception as e:
                        logger.error(f"[{client_id}] Frame {frame_num}: WebSocket send error -> {e}")
                        break

                #   # ---------------- Background Storage ----------------
                # if frame_num % 20 == 0:
                #     annotated_frame = detections.get("Annotated_Frame")
                #     if annotated_frame is not None:
                #         # def store_frame():
                #             try:
                #                 s3_url = upload_to_s3(annotated_frame, frame_num)
                #                 insert_data(detections, s3_url)
                #                 logger.info(f"[{client_id}] Frame {frame_num} stored successfully")
                #             except Exception as e:
                #                 logger.error(f"[{client_id}] Frame {frame_num} store error -> {e}")

                #         # # Run storage async
                #         # storage_executor.submit(store_frame)

                # enqueue storage
                if frame_num % 20 == 0:
                    

                    

                    annotated_frame = detections.get("Annotated_Frame")
                    if annotated_frame is not None:
                        # FULL COPY (avoid race condition)
                        safe_copy = json.loads(json.dumps(detections))

                        store_queue.put((frame_num, annotated_frame, safe_copy))

            else:
                if ws:
                    asyncio.run_coroutine_threadsafe(
                        ws.send_text(json.dumps({"success": False, "message": error})),
                        loop
                    )
                logger.warning(f"[{client_id}] Frame {frame_num}: No detections - {error}")
                break

        except Exception as e:
            logger.exception(f"[{client_id}] Frame {frame_num}: Pipeline error -> {e}")

    cap.release()
    store_queue.put(None)     # stop storage thread
    if client_id in sessions:
        sessions[client_id]["streaming"] = False

    logger.info(f"[{client_id}] Queue Monitoring stopped and resources released")