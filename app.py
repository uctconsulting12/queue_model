from fastapi import FastAPI, WebSocket,File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.websocket.queue_w_local1 import run_queuemonitoring_detection



from src.handlers.queue_handler import queue_websocket_handler


from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Session Stores --------------
queue_sessions = {}
people_sessions = {}


detection_executor = ThreadPoolExecutor(max_workers=10)
storage_executor = ThreadPoolExecutor(max_workers=5)


#--------------------------------------------------------------------------- WebSocket for all Models ------------------------------------------------------------------------------#



# ---------------- Queue WebSocket ----------------
@app.websocket("/ws/queue/{client_id}")
async def websocket_queue(ws: WebSocket, client_id: str):
    await queue_websocket_handler(detection_executor,storage_executor,ws, client_id, queue_sessions, run_queuemonitoring_detection, "Queue")

