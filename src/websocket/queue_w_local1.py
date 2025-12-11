import cv2
import json
import time
import asyncio
import logging
import sys
import os
import base64
import numpy as np
from multiprocessing import Process, Queue

# Add <project_root>/src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.local_models.queue_model.db_manager import get_camera_config, check_for_updates
from src.models.queue_local import queue_monitering

logger = logging.getLogger("people_counting")
logger.setLevel(logging.INFO)
CHECK_UPDATE_INTERVAL = 120


# ---------------------------------------------------------
# Utility: Frame <-> Base64
# ---------------------------------------------------------

def frame_to_b64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode frame")
    return base64.b64encode(buf).decode("utf-8")


def b64_to_frame(b64_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------
# MULTIPROCESSING STORAGE WORKER
# ---------------------------------------------------------

def run_storage_worker(q, client_id):
    """
    Runs in a SEPARATE PROCESS.
    Handles S3 upload + DB insert.
    """

    # Import inside process (IMPORTANT)
    from src.store_s3.queue_store import upload_to_s3
    from src.database.queue_query import insert_data

    logger.info(f"[{client_id}] Storage worker started.")

    while True:
        item = q.get()

        # Sentinel: exit
        if item is None:
            break

        frame_id, annotated_frame, detections = item

        try:
            # Upload to S3
            s3_url = upload_to_s3(annotated_frame, frame_id)

            # DB insert
            insert_data(detections, s3_url)

            logger.info(f"[{client_id}] Stored frame {frame_id}")

        except Exception as e:
            logger.error(f"[{client_id}] Error storing frame {frame_id}: {e}")

    logger.info(f"[{client_id}] Storage worker exiting...")


# ---------------------------------------------------------
# MAIN DETECTION FUNCTION
# ---------------------------------------------------------

def run_queuemonitoring_detection(
    client_id: str,
    video_url: str,
    camera_id: int,
    user_id: int,
    org_id: int,
    sessions: dict,
    loop: asyncio.AbstractEventLoop,
    storage_executor=None
):
    """
    Runs queue monitoring detection.
    Uses a MULTIPROCESSING storage worker to handle S3 uploads + DB inserts.
    """

    camera_config = get_camera_config(camera_id, force_refresh=False)


    cap = cv2.VideoCapture(video_url)
    print("video_url",video_url)
    if not cap.isOpened():
        logger.error(f"[{client_id}] Unable to open video stream: {video_url}")
        return

    frame_num = 0
    last_update_check = time.time()

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("width-height", width,height)

    for queue in camera_config["queues_coordinates"]:
        rect = queue["rect"]
        
        rect["x"] = int(rect["x"] * width)
        rect["y"] = int(rect["y"] * height)
        rect["w"] = int(rect["w"] * width)
        rect["h"] = int(rect["h"] * height)

    print(camera_config)


    # ---------------------------------------------------------
    # START MULTIPROCESS STORAGE WORKER
    # ---------------------------------------------------------
    store_queue = Queue(maxsize=1000)

    storage_process = Process(
        target=run_storage_worker,
        args=(store_queue, client_id),
        daemon=True
    )
    storage_process.start()

    logger.info(f"[{client_id}] Storage worker process started.")

    # ---------------------------------------------------------
    # PROCESS VIDEO FRAMES
    # ---------------------------------------------------------

    while cap.isOpened() and sessions.get(client_id, {}).get("streaming", False):

        ret, frame = cap.read()
        if not ret:
            continue

        frame_num += 1
        # frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

        # # ------------------ Check for DB updates ------------------
        # current_time = time.time()
        # if current_time - last_update_check > CHECK_UPDATE_INTERVAL:
        #     logger.info("Checking for database updates...")
        #     if check_for_updates(camera_id):
        #         logger.info("Database updates detected! Reloading config...")
        #         camera_config = get_camera_config(camera_id, force_refresh=True)
        #         logger.info("Config reloaded successfully.")
        #     last_update_check = current_time

        # Convert frame to base64
        frame_base64 = frame_to_b64(frame)

        # ------------------ RUN DETECTION -------------------------
        try:
            detections, error = queue_monitering(
                frame_base64, camera_id, user_id, org_id, camera_config
            )

            ws = sessions.get(client_id, {}).get("ws")

            # ------------------ SEND TO CLIENT ---------------------
            if detections:
                payload = {"detections": detections}

                if ws:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            ws.send_text(json.dumps(payload)),
                            loop
                        )
                        logger.info(f"[{client_id}] Frame {frame_num}: Sent to WS")
                    except Exception as e:
                        logger.error(f"[{client_id}] WebSocket send error -> {e}")
                        break

                # ------------------ STORE EVERY 20th FRAME -----------------
                if frame_num % 20 == 0:
                    annotated_frame = detections.get("Annotated_Frame")

                    if annotated_frame is not None:
                        # JSON COPY to avoid race condition
                        safe_copy = json.loads(json.dumps(detections))

                        try:
                            store_queue.put_nowait(
                                (frame_num, annotated_frame, safe_copy)
                            )
                        except:
                            logger.warning(
                                f"[{client_id}] Storage queue full; frame {frame_num} dropped."
                            )

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

    # ---------------------------------------------------------
    # CLEANUP
    # ---------------------------------------------------------
    cap.release()

    # STOP STORAGE PROCESS
    store_queue.put(None)
    storage_process.join(timeout=5)

    if client_id in sessions:
        sessions[client_id]["streaming"] = False

    logger.info(f"[{client_id}] Queue Monitoring stopped & cleaned up.")
