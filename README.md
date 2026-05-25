# Queue Monitoring Model

A real-time queue monitoring service that ingests live camera streams (HLS / Kinesis Video Streams / direct video URLs), runs a YOLO-based detection model on each frame, computes per-queue occupancy/wait metrics inside configured regions of interest (ROIs), and streams the results back to connected clients over WebSockets. Annotated frames and detection records are uploaded to S3 and persisted in a database for later review.

---

## Use Case

The service is designed for retail stores, banks, airports, ticket counters, food courts, and similar environments where operators need visibility into how queues are forming in real time. It can be used to:

- Count the number of people standing inside one or more queue zones per camera.
- Trigger alerts when a queue crosses a configurable threshold.
- Provide live dashboards with annotated video frames.
- Store historical queue data for analytics (peak hours, average wait, staffing decisions).

A single instance can serve multiple cameras and multiple clients concurrently via independent WebSocket sessions.

---

## High-Level Flow

```
                ┌──────────────────────────┐
                │   Client (Browser / UI)  │
                └─────────────┬────────────┘
                              │  WebSocket
                              │  ws://host:8001/ws/queue/{client_id}
                              ▼
   ┌──────────────────────────────────────────────────────┐
   │                FastAPI app (app.py)                  │
   │  ┌────────────────────────────────────────────────┐  │
   │  │ queue_websocket_handler                        │  │
   │  │  • accept connection                           │  │
   │  │  • parse { action: start_stream / stop_stream }│  │
   │  │  • resolve KVS → HLS URL (or pass-through URL) │  │
   │  │  • dispatch detection to ThreadPoolExecutor    │  │
   │  └────────────────────────────────────────────────┘  │
   └──────────────────────────┬───────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────┐
   │   run_queuemonitoring_detection (worker thread)      │
   │   src/websocket/queue_w_local1.py                    │
   │                                                      │
   │   1. Load camera_config (ROIs) from DB               │
   │   2. Open video stream with OpenCV                   │
   │   3. For each frame:                                 │
   │        • encode to base64                            │
   │        • call queue_monitering() → YOLO inference    │
   │        • send detections JSON back over WebSocket    │
   │        • every 20th frame → push to storage queue    │
   └──────────────────────────┬───────────────────────────┘
                              │
                              ▼
   ┌──────────────────────────────────────────────────────┐
   │   Storage worker (separate Process)                  │
   │     • Upload annotated frame to S3                   │
   │     • Insert detection record into DB                │
   └──────────────────────────────────────────────────────┘
```

Key components:

- [app.py](app.py) — FastAPI entrypoint that exposes the `/ws/queue/{client_id}` WebSocket route.
- [src/handlers/queue_handler.py](src/handlers/queue_handler.py) — WebSocket lifecycle: accept, parse messages, start/stop detection.
- [src/websocket/queue_w_local1.py](src/websocket/queue_w_local1.py) — Per-stream detection loop running in a thread pool, plus a multiprocessing storage worker.
- [src/models/queue_local.py](src/models/queue_local.py) — Wraps the local YOLO model (`input_fn` / `predict_fn` / `output_fn`).
- [src/local_models/queue_model/](src/local_models/queue_model/) — Model artifacts and inference logic.
- [src/utils/kvs_stream.py](src/utils/kvs_stream.py) — Resolves an AWS Kinesis Video Stream name into an HLS URL.
- [src/store_s3/queue_store.py](src/store_s3/queue_store.py) — Annotated-frame upload to S3.
- [src/database/queue_query.py](src/database/queue_query.py) — DB insert for detection records.
- [model.pt](model.pt) / [yolov8n.pt](yolov8n.pt) — Model weights.

---

## Installation

### Prerequisites

- Python 3.10+ (the Docker image uses PyTorch 2.5.1 + CUDA 12.1)
- An NVIDIA GPU with CUDA drivers (recommended; CPU works but is slow)
- AWS credentials configured (for KVS, S3, and the DB if it runs on RDS) at `~/.aws/credentials`
- A Postgres-compatible database reachable from the host

### 1. Clone the repository

```bash
git clone <repo-url>
cd queue_model
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you need GPU PyTorch, uncomment the CUDA wheels at the top of [requirements.txt](requirements.txt) before installing.

### 4. Configure environment variables

Create a `.env` file at the project root with at least:

```env
# AWS
AWS_REGION=ap-south-1
S3_BUCKET=<your-bucket>

# Database
DB_HOST=<host>
DB_PORT=5432
DB_NAME=<db>
DB_USER=<user>
DB_PASSWORD=<password>
```

(See `src/database/` and `src/store_s3/` for the exact keys each module expects.)

---

## Running the Service

### Local (uvicorn)

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

The WebSocket endpoint will be available at:

```
ws://localhost:8001/ws/queue/{client_id}
```

### Docker

Build the image:

```bash
docker build -t queue_image .
```

Run with GPU and AWS credentials mounted:

```bash
# Linux / macOS
docker run --gpus all -d \
  -p 8001:8001 \
  -v "$HOME/.aws:/root/.aws" \
  --name queue_container \
  queue_image

# Windows (PowerShell)
docker run --gpus all -d `
  -p 8001:8001 `
  -v "$env:USERPROFILE/.aws:/root/.aws" `
  --name queue_container `
  queue_image
```

---

## Using the WebSocket API

### Start a stream

Send this JSON message after connecting to `ws://host:8001/ws/queue/<client_id>`:

```json
{
  "action": "start_stream",
  "stream_name": "Cam424",
  "camera_id": 10,
  "user_id": 10,
  "org_id": 10,
  "threshold": 80,
  "alert_rate": 90,
  "region": "us-east-1"
}
```

- `stream_name` — KVS stream name, or a direct HLS / `https://...mp4` URL.
- `camera_id` — used to fetch the camera's ROI configuration from the database.
- `region` — AWS region for KVS resolution (defaults to `ap-south-1`).

Example direct video URL for testing:

```
https://ai-search-video.s3.us-east-1.amazonaws.com/ai_search_videos/Vid.mp4
```

### Stop a stream

```json
{ "action": "stop_stream" }
```

### Server response

For each processed frame the server sends:

```json
{
  "detections": {
    "queues": [ ... ],
    "Annotated_Frame": "<base64-jpeg>",
    ...
  }
}
```

If detection fails, the server sends `{ "success": false, "message": "<error>" }` and closes the stream.

---

## Local Testing

A quick smoke test for the inference pipeline (no WebSocket) is available in [local_test.py](local_test.py), and a DB connectivity check in [Check_DB_Local.py](Check_DB_Local.py):

```bash
python local_test.py
python Check_DB_Local.py
```

---

## Project Structure

```
queue_model/
├── app.py                       # FastAPI entrypoint (queue WS route)
├── main.py                      # Legacy multi-model entrypoint (PPE, people, employee, queue)
├── Dockerfile
├── requirements.txt
├── model.pt / yolov8n.pt        # Model weights
├── roi.json                     # Sample ROI definition
├── local_test.py
├── Check_DB_Local.py
└── src/
    ├── handlers/                # WebSocket handlers
    ├── websocket/               # Per-stream detection loops
    ├── models/                  # Model wrappers
    ├── local_models/queue_model # Inference, model loader, DB config manager
    ├── database/                # DB queries
    ├── store_s3/                # S3 uploaders
    ├── utils/                   # KVS helpers, etc.
    └── websocket/roi.json
```

---

## Notes

- Detection runs in a `ThreadPoolExecutor` worker, while S3 uploads + DB writes run in a separate `multiprocessing.Process` so storage I/O does not block inference.
- Only every 20th frame is persisted to S3/DB to keep storage usage manageable; tune this in [src/websocket/queue_w_local1.py](src/websocket/queue_w_local1.py).
- ROI coordinates are stored in normalized form (0–1) and scaled to the actual frame dimensions at runtime.
