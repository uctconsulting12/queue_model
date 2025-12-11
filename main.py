
from fastapi import FastAPI, WebSocket
from handler.ws_handler import websocket_handler
from src.websocket.ppe import run_ppe_detection
from src.websocket.queue import run_queuemonitoring_detection
from src.websocket.people import run_peoplecounting_detection
from src.websocket.employee import run_employeemonitoring_detection

from concurrent.futures import ThreadPoolExecutor

app = FastAPI()
executer=ThreadPoolExecutor(max_workers=10)



# ---------------- Session Stores ----------------
ppe_sessions = {}
queue_sessions = {}
people_sessions = {}
employee_sessions={}



# ---------------- PPE WebSocket ----------------
@app.websocket("/ws/ppe/{client_id}")
async def websocket_ppe(ws: WebSocket, client_id: str):
    await websocket_handler(executer,ws, client_id, ppe_sessions, run_ppe_detection, "PPE")


# ---------------- Queue WebSocket ----------------
@app.websocket("/ws/queue/{client_id}")
async def websocket_queue(ws: WebSocket, client_id: str):
    await websocket_handler(executer,ws, client_id, queue_sessions, run_queuemonitoring_detection, "Queue")


# ---------------- People Counting WebSocket ----------------
@app.websocket("/ws/people_counting/{client_id}")
async def websocket_people(ws: WebSocket, client_id: str):
    await websocket_handler(executer,ws, client_id, people_sessions, run_peoplecounting_detection, "PeopleCounting")

# ---------------- People Counting WebSocket ----------------
@app.websocket("/ws/employee/{client_id}")
async def websocket_people(ws: WebSocket, client_id: str):
    await websocket_handler(executer,ws, client_id, employee_sessions, run_employeemonitoring_detection, "EmployeeMonitoring")