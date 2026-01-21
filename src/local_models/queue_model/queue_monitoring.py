# queue_monitoring.py
"""
Queue monitoring core - same logic as AWS version
Person detection, tracking, queue assignment, wait time calculation
Updated: All waiting times in HH:MM:SS format
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
MAX_TRACKING_DISTANCE = 90.0
STALE_TRACK_SECONDS = 8.0


def seconds_to_hms(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS format

    Args:
        seconds: Time in seconds (float)

    Returns:
        Formatted string "HH:MM:SS"
    """
    if seconds < 0:
        seconds = 0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class SimplePersonTracker:
    """Simple person tracker with ID assignment"""

    def __init__(self):
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.removed_tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Dict[str, Any]]:
        """Update tracker with new detections"""
        current_time = datetime.now()
        updated_tracks = []
        matched_track_ids = set()

        # Match detections to existing tracks
        for x, y, w, h, conf in detections:
            cx, cy = x + w // 2, y + h // 2
            best_match_id = None
            best_distance = MAX_TRACKING_DISTANCE

            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue

                tx, ty = track['center']
                distance = np.sqrt((cx - tx) ** 2 + (cy - ty) ** 2)

                if distance < best_distance:
                    best_distance = distance
                    best_match_id = track_id

            if best_match_id:
                self.tracks[best_match_id].update({
                    'bbox': (x, y, w, h),
                    'center': (cx, cy),
                    'confidence': conf,
                    'last_seen': current_time
                })
                matched_track_ids.add(best_match_id)
                updated_tracks.append({
                    'person_id': best_match_id,
                    'bbox': (x, y, w, h),
                    'confidence': conf,
                    'entry_time': self.tracks[best_match_id].get('entry_time', current_time)
                })
            else:
                person_id = self.next_id
                self.next_id += 1

                self.tracks[person_id] = {
                    'bbox': (x, y, w, h),
                    'center': (cx, cy),
                    'confidence': conf,
                    'entry_time': current_time,
                    'last_seen': current_time
                }

                updated_tracks.append({
                    'person_id': person_id,
                    'bbox': (x, y, w, h),
                    'confidence': conf,
                    'entry_time': current_time
                })

        # Remove stale tracks
        stale_ids = []
        for track_id, track in self.tracks.items():
            time_since_seen = (current_time - track['last_seen']).total_seconds()
            if time_since_seen > STALE_TRACK_SECONDS:
                stale_ids.append(track_id)

        for track_id in stale_ids:
            self.removed_tracks[track_id] = self.tracks.pop(track_id)

        return updated_tracks

    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_tracks": len(self.tracks),
            "total_tracks_created": self.next_id - 1,
            "removed_tracks": len(self.removed_tracks)
        }


class QueueMonitoringSystem:
    """Queue monitoring system - same as AWS version"""

    def __init__(self, model, camera_config: Dict[str, Any]):
        self.model = model
        self.config = camera_config

        # All required fields - no defaults
        if "camid" not in camera_config:
            raise ValueError("camid is required in camera_config")
        if "queues_coordinates" not in camera_config:
            raise ValueError("queues_coordinates is required in camera_config")
        if "max_length_allowed" not in camera_config:
            raise ValueError("max_length_allowed is required in camera_config")
        if "max_waiting_time_queue" not in camera_config:
            raise ValueError("max_waiting_time_queue is required in camera_config")
        if "max_waiting_time_front_person" not in camera_config:
            raise ValueError("max_waiting_time_front_person is required in camera_config")

        self.camid = camera_config["camid"]
        self.queues = camera_config["queues_coordinates"]
        self.max_length = camera_config["max_length_allowed"]
        self.max_queue_wait = camera_config["max_waiting_time_queue"]
        self.max_front_wait = camera_config["max_waiting_time_front_person"]

        self.tracker = SimplePersonTracker()
        self.entry_counters = {q["queue_id"]: 0 for q in self.queues}
        self.exit_counters = {q["queue_id"]: 0 for q in self.queues}
        # Alert debouncing - tracks if alert is currently active for each queue
        self.queue_alert_active = {q["queue_id"]: False for q in self.queues}
        self.frame_count = 0

        logger.info(f"Initialized QueueMonitoringSystem for camera {self.camid} with {len(self.queues)} queues")

    def _point_in_rect(self, px: int, py: int, rect: Dict[str, int]) -> bool:
        """Check if point is inside rectangle"""
        x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']
        return x <= px < (x + w) and y <= py < (y + h)

    def _assign_to_queue(self, person: Dict[str, Any]) -> Optional[int]:
        """Assign person to queue based on position"""
        x, y, w, h = person['bbox']
        cx, cy = x + w // 2, y + h // 2

        for queue in self.queues:
            if self._point_in_rect(cx, cy, queue['rect']):
                return queue['queue_id']

        return None

    def process_frame(self, frame: np.ndarray, return_annotated: bool = True) -> Dict[str, Any]:
        """Process frame - same as AWS version"""
        start_time = time.time()
        self.frame_count += 1
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # YOLO detection
        try:
            results = self.model(frame, verbose=False)
            detections = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Person class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        detections.append((x1, y1, x2 - x1, y2 - y1, conf))
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            detections = []

        # Update tracker
        tracked_persons = self.tracker.update(detections)

        # Assign persons to queues
        queue_assignments = {q["queue_id"]: [] for q in self.queues}

        for person in tracked_persons:
            queue_id = self._assign_to_queue(person)
            if queue_id is not None:
                entry_time = person.get('entry_time')
                if entry_time:
                    wait_time_seconds = (current_time - entry_time).total_seconds()
                    queue_assignments[queue_id].append({
                        'person_id': person['person_id'],
                        'bbox': person['bbox'],
                        'confidence': person['confidence'],
                        'entry_time': entry_time,
                        'wait_time': wait_time_seconds,
                        'wait_time_hms': seconds_to_hms(wait_time_seconds)  # HH:MM:SS format
                    })

        # Build queue results with HH:MM:SS format
        queue_results = []

        for queue in self.queues:
            queue_id = queue["queue_id"]
            queue_name = queue.get("name", f"Queue_{queue_id}")
            persons = queue_assignments[queue_id]
            person_count = len(persons)

            # Calculate waiting times
            if persons:
                wait_times = [p['wait_time'] for p in persons]
                avg_wait_time_seconds = sum(wait_times) / len(wait_times)
                max_wait_time_seconds = max(wait_times)

                # Find front person (assuming lowest Y coordinate is front)
                front_person = min(persons, key=lambda p: p['bbox'][1])
                front_wait_time_seconds = front_person['wait_time']
            else:
                avg_wait_time_seconds = 0.0
                max_wait_time_seconds = 0.0
                front_wait_time_seconds = 0.0

            # Convert to HH:MM:SS format
            avg_wait_time_hms = seconds_to_hms(avg_wait_time_seconds)
            max_wait_time_hms = seconds_to_hms(max_wait_time_seconds)
            front_wait_time_hms = seconds_to_hms(front_wait_time_seconds)

            # Check thresholds
            is_overcrowded = person_count > self.max_length
            waiting_time_exceeded = max_wait_time_seconds > self.max_queue_wait
            front_person_wait_exceeded = front_wait_time_seconds > self.max_front_wait

            # Alert logic with debouncing
            alert_triggered = is_overcrowded or waiting_time_exceeded or front_person_wait_exceeded

            # Update alert state
            if alert_triggered:
                if not self.queue_alert_active[queue_id]:
                    # Alert just became active
                    logger.warning(f"ALERT: Queue {queue_id} - Overcrowded: {is_overcrowded}, "
                                   f"Wait exceeded: {waiting_time_exceeded}, "
                                   f"Front wait exceeded: {front_person_wait_exceeded}")
                    self.queue_alert_active[queue_id] = True
            else:
                if self.queue_alert_active[queue_id]:
                    # Alert just cleared
                    logger.info(f"Alert cleared for queue {queue_id}")
                    self.queue_alert_active[queue_id] = False

            # Build detection list with HH:MM:SS format
            detections = []
            for person in persons:
                detections.append({
                    "person_id": person['person_id'],
                    "bbox": list(person['bbox']),
                    "confidence": round(person['confidence'], 2),
                    "waiting_time": person['wait_time_hms']  # HH:MM:SS format
                })

            queue_result = {
                "queue_id": queue_id,
                "name": queue_name,
                "person_count": person_count,
                "average_waiting_time": avg_wait_time_hms,  # HH:MM:SS format
                "max_waiting_time": max_wait_time_hms,  # HH:MM:SS format
                "front_person_waiting_time": front_wait_time_hms,  # HH:MM:SS format
                "is_overcrowded": is_overcrowded,
                "waiting_time_exceeded": waiting_time_exceeded,
                "front_person_wait_exceeded": front_person_wait_exceeded,
                "alert_active": self.queue_alert_active[queue_id],
                "detections": detections
            }

            queue_results.append(queue_result)

        # Processing time
        processing_time = time.time() - start_time

        # Build result
        result = {
            "timestamp": timestamp,
            "queue_results": queue_results,
            "total_persons_detected": len(tracked_persons),
            "processing_time_ms": round(processing_time * 1000, 2),
            "frame_number": self.frame_count
        }

        # Add annotated frame if requested
        if return_annotated:
            result['Annotated_Frame'] = self._create_annotated_frame(
                frame, tracked_persons, queue_assignments
            )

        return result

    def _create_annotated_frame(self, frame: np.ndarray, tracked_persons: List[Dict[str, Any]],
                                queue_assignments: Dict[int, List[Dict[str, Any]]]) -> np.ndarray:
        """Create annotated frame with queue visualization"""
        try:
            annotated = frame.copy()
            frame_height, frame_width = frame.shape[:2]

            # Helper function to draw text with background
            def draw_text_with_background(img, text, position, font, font_scale, text_color, bg_color, thickness,
                                          padding):
                """Draw text with background rectangle for better visibility"""
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                x, y = position

                # Draw background rectangle
                cv2.rectangle(img,
                              (x - padding, y - text_height - padding),
                              (x + text_width + padding, y + baseline + padding),
                              bg_color, -1)

                # Draw text
                cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # Helper function to find front person in queue
            def find_front_person_in_queue(queue_id):
                """Find the front person in a queue (lowest y-coordinate)"""
                persons = queue_assignments.get(queue_id, [])
                if not persons:
                    return None

                # Front person is the one with lowest y-coordinate (top of frame)
                front_person = min(persons, key=lambda p: p['bbox'][1])
                return front_person['person_id']

            # Helper function to determine queue status color
            def get_queue_status_color(queue_id, count):
                """Determine color based on queue status"""
                is_overcrowded = count > self.max_length

                persons = queue_assignments.get(queue_id, [])
                waiting_time_exceeded = False
                if persons:
                    max_wait = max([p['wait_time'] for p in persons])
                    waiting_time_exceeded = max_wait > self.max_queue_wait

                if is_overcrowded or waiting_time_exceeded:
                    return (0, 0, 255)  # Red - alert
                elif count > 0:
                    return (0, 255, 0)  # Green - occupied
                else:
                    return (128, 128, 128)  # Gray - empty

            # Add semi-transparent overlay for header
            try:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (frame_width, 60), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
            except Exception as e:
                logger.error(f"Error adding header overlay: {e}")

            # Add camera info at top
            try:
                cam_info = f"Camera ID: {self.camid}"
                cv2.putText(annotated, cam_info, (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                logger.error(f"Error adding camera info: {e}")

            # Add queue summary info with better spacing
            try:
                y_offset = 70
                for i, queue in enumerate(self.queues):
                    queue_id = queue['queue_id']
                    queue_name = queue['name']
                    persons = queue_assignments.get(queue_id, [])
                    queue_length = len(persons)

                    # Calculate average wait time in HH:MM:SS format
                    if persons:
                        avg_wait_seconds = sum([p['wait_time'] for p in persons]) / len(persons)
                        avg_wait_hms = seconds_to_hms(avg_wait_seconds)
                    else:
                        avg_wait_hms = "00:00:00"

                    # Color code the status
                    status_color = get_queue_status_color(queue_id, queue_length)

                    # Draw status indicator circle
                    cv2.circle(annotated, (25, y_offset - 8), 8, status_color, -1)
                    cv2.circle(annotated, (25, y_offset - 8), 8, (255, 255, 255), 1)

                    queue_info = f"{queue_name}: {queue_length} people, Avg: {avg_wait_hms}"
                    cv2.putText(annotated, queue_info, (45, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    y_offset += 30
            except Exception as e:
                logger.error(f"Error adding queue summary: {e}")

            # Draw queue rectangles with dynamic colors and queue names
            try:
                for queue in self.queues:
                    rect = queue['rect']
                    queue_id = queue['queue_id']
                    queue_name = queue['name']
                    persons = queue_assignments.get(queue_id, [])
                    count = len(persons)

                    x = max(0, min(frame_width - 1, rect['x']))
                    y = max(0, min(frame_height - 1, rect['y']))
                    w = max(1, min(frame_width - x, rect['w']))
                    h = max(1, min(frame_height - y, rect['h']))

                    color = get_queue_status_color(queue_id, count)
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

                    # Draw queue name label at top-left of queue with background
                    label_y = max(y + 25, 25)
                    draw_text_with_background(annotated, queue_name, (x + 8, label_y),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                                              color, 2, 5)

            except Exception as e:
                logger.error(f"Error drawing queue rectangles: {e}")

            # Find front persons for each queue
            front_persons = {}
            for queue in self.queues:
                queue_id = queue['queue_id']
                front_person = find_front_person_in_queue(queue_id)
                if front_person:
                    front_persons[queue_id] = front_person

            # Draw person bounding boxes with improved text visibility
            try:
                for person in tracked_persons:
                    x, y, w, h = person['bbox']
                    person_id = person['person_id']
                    entry_time = person['entry_time']

                    x1, y1 = x, y
                    x2, y2 = x + w, y + h

                    # Clamp coordinates
                    x1 = max(0, min(frame_width - 1, x1))
                    y1 = max(0, min(frame_height - 1, y1))
                    x2 = max(x1 + 1, min(frame_width, x2))
                    y2 = max(y1 + 1, min(frame_height, y2))

                    # Find which queue this person is in
                    person_queue_id = None
                    for qid, persons in queue_assignments.items():
                        if any(p['person_id'] == person_id for p in persons):
                            person_queue_id = qid
                            break

                    is_inside_queue = person_queue_id is not None
                    is_front_person = person_queue_id in front_persons and front_persons[person_queue_id] == person_id

                    # Determine colors
                    if is_front_person:
                        bbox_color = (0, 0, 255)  # Red for front person
                        text_color = (255, 255, 255)  # White text
                        bg_color = (0, 0, 200)  # Dark red background
                    elif is_inside_queue:
                        bbox_color = (0, 255, 0)  # Green for in queue
                        text_color = (255, 255, 255)  # White text
                        bg_color = (0, 150, 0)  # Dark green background
                    else:
                        bbox_color = (255, 255, 255)  # White for outside
                        text_color = (0, 0, 0)  # Black text
                        bg_color = (200, 200, 200)  # Light gray background

                    # Draw bounding box with thicker line for better visibility
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), bbox_color, 3)

                    # Draw person ID inside box with background
                    id_text = f"ID:{person_id}"
                    text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    id_x = x1 + (x2 - x1 - text_size[0]) // 2
                    id_y = y1 + (y2 - y1 + text_size[1]) // 2

                    # Background for ID
                    padding = 6
                    cv2.rectangle(annotated,
                                  (id_x - padding, id_y - text_size[1] - padding),
                                  (id_x + text_size[0] + padding, id_y + padding),
                                  bg_color, -1)

                    cv2.putText(annotated, id_text, (id_x, id_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

                    # Draw waiting time above box if in queue with background (HH:MM:SS format)
                    if is_inside_queue and entry_time:
                        current_time = datetime.now()
                        wait_time_seconds = (current_time - entry_time).total_seconds()
                        wait_time_hms = seconds_to_hms(wait_time_seconds)

                        if is_front_person:
                            wait_text = f"FRONT: {wait_time_hms}"
                            wait_bg_color = (0, 0, 150)  # Dark red
                            wait_text_color = (255, 255, 255)  # White
                        else:
                            wait_text = f"{wait_time_hms}"
                            wait_bg_color = (0, 100, 0)  # Dark green
                            wait_text_color = (255, 255, 255)  # White

                        # Position above box with enough clearance
                        wait_y = max(30, y1 - 15)
                        draw_text_with_background(annotated, wait_text, (x1, wait_y),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                                  wait_text_color, wait_bg_color, 2, 5)

            except Exception as e:
                logger.error(f"Error drawing person annotations: {e}")

            # Add system info at bottom right with background
            try:
                current_time_str = datetime.now().strftime("%H:%M:%S")
                sys_info = f"Time: {current_time_str}"

                info_x = frame_width - 180
                info_y = frame_height - 20

                draw_text_with_background(annotated, sys_info, (info_x, info_y),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),
                                          (0, 0, 0), 2, 8)
            except Exception as e:
                logger.warning(f"Failed to add system info: {e}")

            return annotated

        except Exception as e:
            logger.error(f"Critical error creating annotated frame: {e}")
            return frame.copy()

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics including alert states"""
        return {
            "camera_id": self.camid,
            "frames_processed": self.frame_count,
            "queues_configured": len(self.queues),
            "tracker_stats": self.tracker.get_stats(),
            "entry_counters": self.entry_counters,
            "exit_counters": self.exit_counters,
            "alert_states": self.queue_alert_active  # Show which queues have active alerts
        }