# queue_monitoring.py
"""
Queue monitoring core - same logic as AWS version
Person detection, tracking, queue assignment, wait time calculation
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
        
        # Queue assignments
        queue_assignments = {queue['queue_id']: [] for queue in self.queues}
        
        person_data = {
            'ids': [], 'queue_ids': [], 'entry_times': [], 
            'wait_times': [], 'bboxes': [], 'confidences': []
        }
        
        for person in tracked_persons:
            queue_id = self._assign_to_queue(person)
            person_id = person['person_id']
            entry_time = person['entry_time']
            wait_time = (current_time - entry_time).total_seconds() / 60.0
            
            person_data['ids'].append(person_id)
            person_data['queue_ids'].append(queue_id if queue_id else 0)
            person_data['entry_times'].append(entry_time.strftime("%H:%M:%S"))
            person_data['wait_times'].append(round(wait_time, 2))
            person_data['bboxes'].append(person['bbox'])
            person_data['confidences'].append(person['confidence'])
            
            if queue_id:
                queue_assignments[queue_id].append({
                    'person_id': person_id,
                    'wait_time': wait_time,
                    'bbox': person['bbox']
                })
        
        # Calculate queue stats
        queue_stats = self._calculate_queue_stats(queue_assignments)
        
        # Create annotated frame
        annotated_frame = None
        if return_annotated:
            annotated_frame = self._create_annotated_frame(
                frame.copy(), tracked_persons, queue_assignments
            )
        
        processing_time = time.time() - start_time
        
        # Build result
        result = {
            "Frame_Id": str(int(time.time() * 1000)),
            "Time_stamp": timestamp,
            "Queue_Count": len(self.queues),
            "Queue_Name": [q["name"] for q in self.queues],
            "Queue_Length": queue_stats['lengths'],
            "Front_person_Wt": queue_stats['front_wait_times'],
            "Average_wt_time": queue_stats['avg_wait_times'],
            "Status": queue_stats['statuses'],
            "Should_Alert": queue_stats['should_alert'],  # NEW: Alert debouncing flag
            "Total_people_detected": len(tracked_persons),
            "People_ids": person_data['ids'],
            "Queue_Assignment": person_data['queue_ids'],
            "Entry_time": person_data['entry_times'],
            "People_wt_time": person_data['wait_times'],
            "Annotated_Frame": annotated_frame,
            "x": [bbox[0] for bbox in person_data['bboxes']],
            "y": [bbox[1] for bbox in person_data['bboxes']],
            "w": [bbox[2] for bbox in person_data['bboxes']],
            "h": [bbox[3] for bbox in person_data['bboxes']],
            "accuracy": person_data['confidences'],
            "Entry_Counts": [self.entry_counters[q["queue_id"]] for q in self.queues],
            "Exit_Counts": [self.exit_counters[q["queue_id"]] for q in self.queues],
            "Processing_Time_ms": int(processing_time * 1000),
            "Tracker_Stats": self.tracker.get_stats()
        }
        
        return result

    def _calculate_queue_stats(self, queue_assignments: Dict[int, List]) -> Dict[str, List]:
        """
        Calculate queue statistics with alert debouncing
        
        Returns actual status AND whether a new alert should be triggered.
        Alert is only triggered once when problem first detected, and resets when significantly resolved.
        
        Status field behavior:
        - If Should_Alert = true: Shows the problem reason (e.g., "QUEUE_TOO_LONG")
        - If Should_Alert = false: Shows empty string ""
        
        Reset thresholds (need significant improvement):
        - Queue length: Drops to 70% of max_length or below
        - Front wait: Drops to 70% of max_front_wait or below
        - Avg wait: Drops to 70% of max_queue_wait or below
        """
        lengths = []
        front_wait_times = []
        avg_wait_times = []
        statuses = []
        alert_states = []  # NEW: Tracks if new alert should be sent
        
        # Reset thresholds (70% of max values for significant improvement)
        reset_length_threshold = int(self.max_length * 0.7)
        reset_front_wait_threshold = (self.max_front_wait / 60.0) * 0.7  # in minutes
        reset_avg_wait_threshold = (self.max_queue_wait / 60.0) * 0.7    # in minutes

        for queue in self.queues:
            queue_id = queue['queue_id']
            persons = queue_assignments.get(queue_id, [])
            length = len(persons)
            lengths.append(length)

            if length == 0:
                # Queue is empty - reset everything
                front_wait_times.append(0.0)
                avg_wait_times.append(0.0)
                statuses.append("")  # Empty string when no alert
                alert_states.append(False)
                self.queue_alert_active[queue_id] = False  # Reset alert state
            else:
                # Calculate wait times
                persons_sorted = sorted(persons, key=lambda p: p['bbox'][1])
                front_person = persons_sorted[0]
                front_wait = front_person['wait_time']
                front_wait_times.append(round(front_wait, 2))

                avg_wait = sum(p['wait_time'] for p in persons) / length
                avg_wait_times.append(round(avg_wait, 2))

                # Determine ACTUAL status (always reflects current reality)
                actual_status = None
                if length > self.max_length:
                    actual_status = "QUEUE_TOO_LONG"
                elif front_wait > (self.max_front_wait / 60.0):
                    actual_status = "FRONT_WAIT_EXCEEDED"
                elif avg_wait > (self.max_queue_wait / 60.0):
                    actual_status = "AVG_WAIT_EXCEEDED"
                else:
                    actual_status = "OK"

                # Determine if NEW alert should be triggered
                should_alert = False
                
                if actual_status != "OK" and not self.queue_alert_active[queue_id]:
                    # Problem detected AND no active alert - TRIGGER NEW ALERT
                    should_alert = True
                    self.queue_alert_active[queue_id] = True
                    logger.info(f"🚨 Alert triggered for Queue {queue_id}: {actual_status}")
                    # Show status reason when alerting
                    statuses.append(actual_status)
                    
                elif actual_status == "OK" and self.queue_alert_active[queue_id]:
                    # Check if problem is SIGNIFICANTLY resolved (70% threshold)
                    # This ensures we only reset after real improvement, not just 1 person leaving
                    is_significantly_resolved = (
                        length <= reset_length_threshold and
                        front_wait <= reset_front_wait_threshold and
                        avg_wait <= reset_avg_wait_threshold
                    )
                    
                    if is_significantly_resolved:
                        # Problem SIGNIFICANTLY resolved - RESET alert state
                        self.queue_alert_active[queue_id] = False
                        logger.info(f"✅ Alert cleared for Queue {queue_id} - Significantly improved")
                    
                    # Status is empty when no alert needed
                    statuses.append("")
                    
                else:
                    # Either: problem persists (alert already active) OR no problem at all
                    # In both cases: Status is empty (no new alert)
                    statuses.append("")

                alert_states.append(should_alert)

        return {
            'lengths': lengths,
            'front_wait_times': front_wait_times,
            'avg_wait_times': avg_wait_times,
            'statuses': statuses,           # Shows reason ONLY when Should_Alert=true, else ""
            'should_alert': alert_states     # Only True when NEW alert should be sent
        }

    def _create_annotated_frame(self, frame: np.ndarray, tracked_persons: List,
                                queue_assignments: Dict) -> np.ndarray:
        """Create enhanced annotated frame with improved text visibility - No overlapping"""
        try:
            annotated = frame.copy()
            frame_height, frame_width = annotated.shape[:2]

            # Helper: Draw text with background for better visibility
            def draw_text_with_background(img, text, position, font, font_scale, text_color, bg_color, thickness=2, padding=5):
                """Draw text with a solid background to prevent overlapping"""
                x, y = position
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_w, text_h = text_size

                # Draw background rectangle
                cv2.rectangle(img,
                            (x - padding, y - text_h - padding),
                            (x + text_w + padding, y + padding),
                            bg_color, -1)

                # Draw text
                cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

                return text_w, text_h

            # Helper: Find front person in each queue (earliest entry time)
            def find_front_person_in_queue(queue_id: int) -> Optional[int]:
                """Find the person who entered the queue first"""
                queue_people = []
                for person in tracked_persons:
                    for qid, persons in queue_assignments.items():
                        if qid == queue_id and any(p['person_id'] == person['person_id'] for p in persons):
                            queue_people.append((person['person_id'], person['entry_time']))
                            break

                if queue_people:
                    queue_people.sort(key=lambda x: x[1])
                    return queue_people[0][0]
                return None

            # Helper: Get queue status color
            def get_queue_status_color(queue_id: int, count: int) -> tuple:
                """Determine queue color based on status"""
                persons = queue_assignments.get(queue_id, [])
                if not persons:
                    return (0, 255, 0)

                front_wait = max([p['wait_time'] for p in persons]) if persons else 0
                avg_wait = sum([p['wait_time'] for p in persons]) / len(persons) if persons else 0

                if count > self.max_length:
                    return (0, 0, 255)
                elif front_wait > (self.max_front_wait / 60.0) * 0.8:
                    return (0, 165, 255)
                elif avg_wait > (self.max_queue_wait / 60.0) * 0.8:
                    return (0, 165, 255)
                else:
                    return (0, 255, 0)

            # Create semi-transparent overlay for top info panel
            overlay = annotated.copy()

            # Calculate height needed for top panel
            num_queues = len(self.queues)
            panel_height = 50 + (num_queues * 30)

            # Draw top panel background (semi-transparent black)
            cv2.rectangle(overlay, (0, 0), (frame_width, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

            # Add Camera ID at top with better spacing
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
                    avg_wait = sum([p['wait_time'] for p in persons]) / len(persons) if persons else 0

                    # Color code the status
                    status_color = get_queue_status_color(queue_id, queue_length)

                    # Draw status indicator circle
                    cv2.circle(annotated, (25, y_offset - 8), 8, status_color, -1)
                    cv2.circle(annotated, (25, y_offset - 8), 8, (255, 255, 255), 1)

                    queue_info = f"{queue_name}: {queue_length} people, Avg: {avg_wait:.1f}min"
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

                    # Draw waiting time above box if in queue with background
                    if is_inside_queue and entry_time:
                        current_time = datetime.now()
                        wait_time_seconds = (current_time - entry_time).total_seconds()
                        wait_time_minutes = int(wait_time_seconds / 60)

                        if is_front_person:
                            wait_text = f"FRONT: {wait_time_minutes}min"
                            wait_bg_color = (0, 0, 150)  # Dark red
                            wait_text_color = (255, 255, 255)  # White
                        else:
                            wait_text = f"{wait_time_minutes}min"
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
            "alert_states": self.queue_alert_active  # NEW: Show which queues have active alerts
        }
