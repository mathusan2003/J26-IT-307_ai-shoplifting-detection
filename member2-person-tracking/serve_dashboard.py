"""Serve the Component 2 React dashboard and live tracking API."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dashboard" / "ui_build"


class ControlIn(BaseModel):
    action: str
    source: Optional[str] = None
    person_id: Optional[int] = Field(default=None)


def create_app(session=None):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    from src.behaviour_bridge import empty_behaviour_state
    from src.live_session import get_session
    from src.object_bridge import empty_object_state

    app = FastAPI(title="Member 2 Tracking Dashboard")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    live = session if session is not None else get_session()

    @app.get("/api/sources")
    def sources():
        return {"weights": str(live.weights), "videos": live.list_sources()}

    @app.get("/api/state")
    def state():
        return live.snapshot()

    @app.get("/api/behaviour")
    def behaviour():
        payload = live.snapshot()
        return payload.get("behaviour") or empty_behaviour_state()

    @app.get("/api/objects")
    def objects():
        payload = live.snapshot()
        return payload.get("objects") or empty_object_state()

    @app.post("/api/control")
    def control(payload: ControlIn):
        return live.control(payload.action, payload.source, payload.person_id)

    @app.get("/stream.mjpg")
    def stream():
        boundary = b"--frame"

        def generate():
            import time

            while True:
                jpeg = live.mjpeg_bytes()
                if jpeg:
                    yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.04)

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stream_behaviour.mjpg")
    def stream_behaviour():
        boundary = b"--frame"

        def generate():
            import time

            while True:
                jpeg = live.mjpeg_behaviour_bytes()
                if jpeg:
                    yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.04)

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/stream_objects.mjpg")
    def stream_objects():
        boundary = b"--frame"

        def generate():
            import time

            while True:
                jpeg = live.mjpeg_objects_bytes()
                if jpeg:
                    yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.04)

        return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

    if DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            candidate = DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            index = DIST / "index.html"
            if index.is_file():
                return FileResponse(index, headers={"Cache-Control": "no-store"})
            return Response("Dashboard build missing. Run npm run build in dashboard/", status_code=404)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Component 2 tracking dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
