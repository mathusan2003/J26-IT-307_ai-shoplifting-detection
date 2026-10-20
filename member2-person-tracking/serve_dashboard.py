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


def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    from src.live_session import get_session

    app = FastAPI(title="Member 2 Tracking Dashboard")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    session = get_session()

    @app.get("/api/sources")
    def sources():
        return {"weights": str(session.weights), "videos": session.list_sources()}

    @app.get("/api/state")
    def state():
        return session.snapshot()

    @app.post("/api/control")
    def control(payload: ControlIn):
        return session.control(payload.action, payload.source, payload.person_id)

    @app.get("/stream.mjpg")
    def stream():
        boundary = b"--frame"

        def generate():
            import time

            while True:
                jpeg = session.mjpeg_bytes()
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
                return FileResponse(index)
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
