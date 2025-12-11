
# import json
# import logging
# import psycopg2
# import os
# from dotenv import load_dotenv

# logger = logging.getLogger("detection")
# logger.setLevel(logging.INFO)

# # Load environment variables from .env
# load_dotenv()


# def get_connection():
#     """Create and return a new PostgreSQL connection."""
#     return psycopg2.connect(
#         host=os.environ["DB_HOST"],
#         dbname=os.environ["DB_NAME"],
#         user=os.environ["DB_USER"],
#         password=os.environ["DB_PASSWORD"],
#         port=int(os.environ.get("DB_PORT", 5432))
#     )


# def insert_data(data, s3_url):
#     """Insert data into queue_monitoring table safely."""
#     conn = None
#     try:
#         conn = get_connection()
#         conn.autocommit = True

#         with conn.cursor() as cursor:
#             insert_query = """
#                 INSERT INTO queue_monitoring (
#                     camid, userid, org_id, frame_id, time_stamp, queue_count,
#                     queue_name, queue_length, front_person_wt, average_wt_time,
#                     status, total_people_detected, people_ids, queue_assignment,
#                     entry_time, people_wt_time, processing_status, x, y, w, h, accuracy, s3_url
#                 ) VALUES (
#                     %s, %s, %s, %s, %s, %s,
#                     %s, %s, %s, %s,
#                     %s, %s, %s, %s,
#                     %s, %s, %s,
#                     %s, %s, %s, %s, %s, %s
#                 )
#                 ON CONFLICT (frame_id) DO UPDATE SET
#                     time_stamp = EXCLUDED.time_stamp,
#                     total_people_detected = EXCLUDED.total_people_detected,
#                     processing_status = EXCLUDED.processing_status,
#                     x = EXCLUDED.x,
#                     y = EXCLUDED.y,
#                     w = EXCLUDED.w,
#                     h = EXCLUDED.h,
#                     accuracy = EXCLUDED.accuracy,
#                     s3_url = EXCLUDED.s3_url;
#             """

#             cursor.execute(insert_query, (
#                 data['camid'],
#                 data['userid'],
#                 data['org_id'],
#                 data['Frame_Id'],
#                 data['Time_stamp'],
#                 data['Queue_Count'],
#                 json.dumps(data['Queue_Name']),
#                 json.dumps(data['Queue_Length']),
#                 json.dumps(data['Front_person_Wt']),
#                 json.dumps(data['Average_wt_time']),
#                 json.dumps(data['Status']),
#                 data['Total_people_detected'],
#                 json.dumps(data['People_ids']),
#                 json.dumps(data['Queue_Assignment']),
#                 json.dumps(data['Entry_time']),
#                 json.dumps(data['People_wt_time']),
#                 data['Processing_Status'],
#                 json.dumps(data['x']),
#                 json.dumps(data['y']),
#                 json.dumps(data['w']),
#                 json.dumps(data['h']),
#                 json.dumps(data['accuracy']),
#                 s3_url
#             ))
#         logger.info(f"✅ Data inserted for frame: {data['Frame_Id']}")
#         return True

#     except Exception as e:
#         logger.error(f"❌ Failed to insert data: {e}")
#         return False

#     finally:
#         if conn:
#             conn.close()





import os
import logging
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import json
import logging

load_dotenv()
logger = logging.getLogger("detection")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

try:
    pool = SimpleConnectionPool(
        minconn=1,        
        maxconn=15,       
        **DB_CONFIG
    )
    logger.info("✅ PostgreSQL Connection Pool Created")
except Exception as e:
    logger.error(f"❌ Error creating connection pool: {e}")
    raise




logger = logging.getLogger("detection")


def insert_data(data, s3_url):
    """Insert data into queue_monitoring table using connection pooling."""

    conn = None
    try:
        conn = pool.getconn()   # ⬅ Borrow connection from pool
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO queue_monitoring (
                camid, userid, org_id, frame_id, time_stamp, queue_count,
                queue_name, queue_length, front_person_wt, average_wt_time,
                status, total_people_detected, people_ids, queue_assignment,
                entry_time, people_wt_time, processing_status, x, y, w, h,
                accuracy, s3_url
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s
            )
            ON CONFLICT (frame_id) DO UPDATE SET
                time_stamp = EXCLUDED.time_stamp,
                total_people_detected = EXCLUDED.total_people_detected,
                processing_status = EXCLUDED.processing_status,
                x = EXCLUDED.x,
                y = EXCLUDED.y,
                w = EXCLUDED.w,
                h = EXCLUDED.h,
                accuracy = EXCLUDED.accuracy,
                s3_url = EXCLUDED.s3_url;
        """

        cursor.execute(insert_query, (
            data['camid'],
            data['userid'],
            data['org_id'],
            data['Frame_Id'],
            data['Time_stamp'],
            data['Queue_Count'],
            json.dumps(data['Queue_Name']),
            json.dumps(data['Queue_Length']),
            json.dumps(data['Front_person_Wt']),
            json.dumps(data['Average_wt_time']),
            json.dumps(data['Status']),
            data['Total_people_detected'],
            json.dumps(data['People_ids']),
            json.dumps(data['Queue_Assignment']),
            json.dumps(data['Entry_time']),
            json.dumps(data['People_wt_time']),
            data['Processing_Status'],
            json.dumps(data['x']),
            json.dumps(data['y']),
            json.dumps(data['w']),
            json.dumps(data['h']),
            json.dumps(data['accuracy']),
            s3_url
        ))

        conn.commit()
        cursor.close()

        logger.info(f"✅ Data inserted for frame: {data['Frame_Id']}")
        return True

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"❌ Failed to insert data: {e}")
        return False

    finally:
        if conn:
            pool.putconn(conn)   # ⬅ Return connection back to pool

