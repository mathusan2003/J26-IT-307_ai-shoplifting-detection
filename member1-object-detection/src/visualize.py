"""Draw Member 1 detections. Boxes come only from model output."""
from __future__ import annotations

from typing import Any

from object_detector import PROTOTYPE_NOTE


def colour_for_class(class_id: int) -> tuple[int, int, int]:
    return ((37 * class_id) % 256, (17 * class_id + 80) % 256, (73 * class_id + 40) % 256)


def draw_detections(
    frame: Any,
    result: dict[str, Any] | None,
    *,
    source: str | None = None,
    infer_fps: float | None = None,
) -> Any:
    """Paint real boxes onto a copy of the frame. Empty results stay empty."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python-headless to draw object overlays.") from exc

    shown = frame.copy()
    payload = result or {}
    detections = list(payload.get("detections") or [])
    for item in detections:
        box = item.get("bbox") or {}
        x1, y1 = int(box.get("x1", 0)), int(box.get("y1", 0))
        x2, y2 = int(box.get("x2", 0)), int(box.get("y2", 0))
        class_id = int(item.get("class_id") or 0)
        colour = colour_for_class(class_id)
        label = f"{item.get('class_name', 'object')} {float(item.get('confidence') or 0):.2f}"
        cv2.rectangle(shown, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(shown, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)

    fps_text = f"{infer_fps:.1f}" if infer_fps is not None else "—"
    hud = (
        f"f={payload.get('frame_id', 0)} n={len(detections)} det={fps_text} "
        f"model={payload.get('model') or 'unloaded'} src={source or '—'}"
    )
    cv2.putText(shown, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        shown,
        "Member 1 pretrained YOLO prototype",
        (10, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not detections:
        message = payload.get("error") or "No objects detected in this frame"
        cv2.putText(shown, message, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 2, cv2.LINE_AA)
    else:
        cv2.putText(
            shown,
            PROTOTYPE_NOTE[:72],
            (10, shown.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return shown
