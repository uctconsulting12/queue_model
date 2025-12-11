# db_manager.py
"""
Database manager - Fetches from PostgreSQL, caches to roi.json
Syncs roi.json with database changes
"""

import json
import logging
import psycopg2
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Database configuration
DB_CONFIG = {
    'host': '54.225.63.242',
    'port': 5432,
    'database': 'visco',
    'user': 'visco_cctv',
    'password': 'Visco@0408'
}

TABLE_NAME = "all_camera_coordinate"
ROI_CACHE_FILE = "roi.json"


def _get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            connect_timeout=10
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


def _parse_coordinate_json(coord_value: Any) -> List[Dict[str, Any]]:
    """Parse coordinate column from database"""
    if coord_value is None:
        return []
    
    if isinstance(coord_value, str):
        coord_value = coord_value.strip()
        if coord_value == "" or coord_value.lower() in ("null", "none"):
            return []
        try:
            coord_value = json.loads(coord_value)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse coordinate JSON: {e}")
            return []
    
    if isinstance(coord_value, list):
        return coord_value
    elif isinstance(coord_value, dict):
        return [coord_value]
    else:
        return []


def _convert_db_to_queue_format(db_queues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert database format to queue monitoring format"""
    converted_queues = []
    
    for queue in db_queues:
        if not isinstance(queue, dict):
            continue
        
        queue_id = queue.get("queue_id", 1)
        name = queue.get("name", f"Queue_{queue_id}")
        
        # Handle different coordinate key names
        coords = queue.get("coordinates") or queue.get("rect") or queue.get("coordinate") or {}

        print(coords)
        
        if not isinstance(coords, dict):
            logger.warning(f"Invalid coordinates for queue {queue_id}")
            continue
        
        try:
            x = round(float(coords.get("x", 0)),3)
            y = round(float(coords.get("y", 0)),3)
            w = round(float(coords.get("w", coords.get("width", 0))),3)
            h = round(float(coords.get("h", coords.get("height", 0))),3)
            
            # if w <= 0 or h <= 0:
            #     logger.warning(f"Invalid dimensions for queue {queue_id}: w={w}, h={h}")
            #     continue
            
            converted_queue = {
                "queue_id": queue_id,
                "name": name,
                "rect": {
                    "x": max(0, x),
                    "y": max(0, y),
                    "w": w,
                    "h": h
                }
            }
            converted_queues.append(converted_queue)
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error converting coordinates for queue {queue_id}: {e}")
            continue
    
    return converted_queues


def fetch_camera_from_db(camera_id: int) -> Optional[Dict[str, Any]]:
    """Fetch camera configuration from PostgreSQL database"""
    conn = None
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        
        query = f"""
        SELECT camera_id, user_id, org_id, region_name, number_of_queues,
               max_length_allowed, max_waiting_time_queue, max_waiting_time_front_person,
               coordinate, created_at, updated_at
        FROM {TABLE_NAME}
        WHERE camera_id = %s
        """
        
        cursor.execute(query, (camera_id,))
        row = cursor.fetchone()
        
        if not row:
            logger.warning(f"Camera {camera_id} not found in database")
            return None
        
        columns = [desc[0] for desc in cursor.description]
        row_dict = dict(zip(columns, row))
        
        # Parse coordinates
        raw_coordinates = _parse_coordinate_json(row_dict.get("coordinate"))
        queues_coordinates = _convert_db_to_queue_format(raw_coordinates)
        
        # Build config
        config = {
            "camid": int(row_dict.get("camera_id", camera_id)),
            "userid": int(row_dict.get("user_id", 0)),
            "org_id": int(row_dict.get("org_id", 0)),
            "region_name": str(row_dict.get("region_name", "")),
            "number_of_queues": int(row_dict.get("number_of_queues", len(queues_coordinates))),
            "max_length_allowed": int(row_dict.get("max_length_allowed", 10)),
            "max_waiting_time_queue": int(row_dict.get("max_waiting_time_queue", 300)),
            "max_waiting_time_front_person": int(row_dict.get("max_waiting_time_front_person", 120)),
            "queues_coordinates": queues_coordinates,
            "created_at": str(row_dict.get("created_at", "")),
            "updated_at": str(row_dict.get("updated_at", "")),
            "fetched_at": datetime.now().isoformat()
        }
        
        logger.info(f"Fetched camera {camera_id} from database")
        return config
        
    except Exception as e:
        logger.error(f"Error fetching camera {camera_id} from database: {e}")
        return None
    finally:
        if conn:
            cursor.close()
            conn.close()


def load_roi_cache() -> Dict[int, Dict[str, Any]]:
    """Load ROI cache from roi.json file"""
    try:
        with open(ROI_CACHE_FILE, 'r') as f:
            data = json.load(f)
            # Convert string keys to integers
            return {int(k): v for k, v in data.items()}
    except FileNotFoundError:
        logger.info(f"{ROI_CACHE_FILE} not found, creating empty cache")
        return {}
    except Exception as e:
        logger.error(f"Error loading ROI cache: {e}")
        return {}


def save_roi_cache(cache: Dict[int, Dict[str, Any]]):
    """Save ROI cache to roi.json file"""
    try:
        with open(ROI_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        logger.info(f"Saved {len(cache)} cameras to {ROI_CACHE_FILE}")
    except Exception as e:
        logger.error(f"Error saving ROI cache: {e}")


def get_camera_config(camera_id: int,force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    """
    Get camera configuration - checks cache first, then database
    
    Args:
        camera_id: Camera ID
        force_refresh: Force fetch from database
    
    Returns:
        Camera configuration dict or None
    """
    # Load cache
    cache = load_roi_cache()
    
    # Check cache if not forcing refresh
    if not force_refresh and camera_id in cache:
        logger.info(f"Camera {camera_id} loaded from cache")
        return cache[camera_id]
    
    # Fetch from database
    config = fetch_camera_from_db(camera_id)
    
    if config:
        # Update cache
        cache[camera_id] = config
        save_roi_cache(cache)
        logger.info(f"Camera {camera_id} fetched from database and cached")
    
    return config


def check_for_updates(camera_id: int) -> bool:
    """
    Check if database has updates for camera
    
    Returns:
        True if updates found, False otherwise
    """
    cache = load_roi_cache()
    
    if camera_id not in cache:
        return True  # Not in cache, need to fetch
    
    # Fetch fresh from database
    db_config = fetch_camera_from_db(camera_id)
    
    if not db_config:
        return False
    
    cached_config = cache[camera_id]
    
    # Compare updated_at timestamps
    db_updated = db_config.get("updated_at", "")
    cache_updated = cached_config.get("updated_at", "")
    
    if db_updated != cache_updated:
        logger.info(f"Updates detected for camera {camera_id}")
        # Update cache
        cache[camera_id] = db_config
        save_roi_cache(cache)
        return True
    
    return False


def get_all_camera_ids() -> List[int]:
    """Get all camera IDs from database"""
    conn = None
    try:
        conn = _get_db_connection()
        cursor = conn.cursor()
        
        query = f"SELECT DISTINCT camera_id FROM {TABLE_NAME}"
        cursor.execute(query)
        
        camera_ids = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(camera_ids)} cameras in database")
        return camera_ids
        
    except Exception as e:
        logger.error(f"Error fetching camera IDs: {e}")
        return []
    finally:
        if conn:
            cursor.close()
            conn.close()


def sync_all_cameras():
    """Sync all cameras from database to roi.json"""
    logger.info("Syncing all cameras from database...")
    
    camera_ids = get_all_camera_ids()
    cache = {}
    
    for camera_id in camera_ids:
        config = fetch_camera_from_db(camera_id)
        if config:
            cache[camera_id] = config
    
    save_roi_cache(cache)
    logger.info(f"Synced {len(cache)} cameras to {ROI_CACHE_FILE}")


if __name__ == "__main__":
    # Test database connectivity
    logger.info("Testing database connection...")
    
    try:
        # Test fetch
        config = fetch_camera_from_db(1)
        if config:
            logger.info(f"✓ Successfully fetched camera 1: {config['region_name']}")
        
        # Test cache
        save_roi_cache({1: config})
        loaded = load_roi_cache()
        logger.info(f"✓ Cache works: {len(loaded)} cameras")
        
        # Test sync
        sync_all_cameras()
        logger.info("✓ All tests passed")
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
