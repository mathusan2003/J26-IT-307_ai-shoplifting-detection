"""Safeguards for behaviour annotations and the comparison runner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from annotation_guard import (
    COMPARISON_APPROACHES,
    AnnotationRejected,
    compare_behaviour_approaches,
    load_behaviour_annotations,
    validate_annotation,
)


class AnnotationGuardTests(unittest.TestCase):
    def test_empty_and_incomplete_inputs_are_rejected(self) -> None:
        with self.assertRaises(AnnotationRejected):
            validate_annotation(None)
        with self.assertRaises(AnnotationRejected):
            validate_annotation({})
        with self.assertRaises(AnnotationRejected):
            validate_annotation(_reviewed(loitering=""))
        with self.assertRaises(AnnotationRejected):
            validate_annotation(_reviewed(review_status="unreviewed"))
        with self.assertRaises(AnnotationRejected):
            load_behaviour_annotations(MODULE_ROOT / "datasets" / "annotation" / "labels")

    def test_folder_names_are_rejected_as_behaviour_targets(self) -> None:
        with self.assertRaises(AnnotationRejected):
            validate_annotation({"label": "shoplifting"})
        with self.assertRaises(AnnotationRejected):
            validate_annotation({"behaviour_target": "normal"})
        with self.assertRaises(AnnotationRejected):
            validate_annotation(_reviewed(loitering="normal"))
        with self.assertRaises(AnnotationRejected):
            validate_annotation(_reviewed(abnormal_movement="shoplifting"))
        accepted = validate_annotation(_reviewed())
        self.assertEqual(accepted["folder_label"], "normal")
        self.assertEqual(accepted["checklist"]["loitering"], "absent")

    def test_present_requires_a_timed_visible_segment(self) -> None:
        with self.assertRaises(AnnotationRejected):
            validate_annotation(_reviewed(loitering="present"))
        accepted = validate_annotation(
            _reviewed(
                loitering="present",
                segments=[
                    {
                        "behaviour": "loitering",
                        "start_s": 1,
                        "end_s": 4,
                        "person": "blue shirt",
                        "what_was_seen": "stood beside the table",
                        "confidence": "high",
                    }
                ],
            )
        )
        self.assertEqual(accepted["checklist"]["loitering"], "present")

    def test_comparison_is_blocked_without_reviewed_labels(self) -> None:
        for annotations in (None, [], [{}]):
            with self.assertRaises(AnnotationRejected):
                compare_behaviour_approaches(annotations)
        result = compare_behaviour_approaches([_reviewed()])
        self.assertFalse(result["evaluated"])
        self.assertIsNone(result["metrics"])
        self.assertEqual(tuple(result["approaches"]), COMPARISON_APPROACHES)
        self.assertEqual(result["reviewed_count"], 1)

    def test_directory_without_json_is_rejected(self) -> None:
        folder = MODULE_ROOT / "outputs" / "empty-annotation-check"
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / "note.txt"
        marker.write_text("not an annotation\n", encoding="utf-8")
        try:
            with self.assertRaises(AnnotationRejected):
                load_behaviour_annotations(folder)
        finally:
            marker.unlink(missing_ok=True)
            folder.rmdir()


def _reviewed(**overrides: object) -> dict:
    checklist = {
        "loitering": "absent",
        "repeated_shelf_visit": "absent",
        "abnormal_movement": "absent",
        "object_interaction": "absent",
        "possible_concealment": "absent",
    }
    segments: list = []
    review_status = "reviewed"
    for key, value in overrides.items():
        if key in checklist:
            checklist[key] = value
        elif key == "segments":
            segments = value  # type: ignore[assignment]
        elif key == "review_status":
            review_status = str(value)
    return {
        "schema_version": 1,
        "video_id": "normal/normal-1.mp4",
        "folder_label": "normal",
        "review_status": review_status,
        "checklist": checklist,
        "segments": segments,
    }


if __name__ == "__main__":
    unittest.main()
