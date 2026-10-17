"""Tests for the real-video split and the Normal versus Shoplifting baseline."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_ROOT / "src"))

from train_video_baseline import train_from_features
from video_dataset import (
    BEHAVIOUR_TYPES,
    CATALOG_PATH,
    FEATURES_PATH,
    SPLIT_PATH,
    VIDEO_ROOT,
    assign_splits,
    file_sha256,
    label_from_folder,
    split_sizes,
)

METRICS_PATH = MODULE_ROOT / "evaluation" / "video_baseline_metrics.json"


class VideoSplitTests(unittest.TestCase):
    def test_folder_name_is_the_only_label(self) -> None:
        self.assertEqual(label_from_folder(Path("normal/normal-1.mp4")), "normal")
        self.assertEqual(label_from_folder(Path("shoplifting/shoplifting-1.mp4")), "shoplifting")
        with self.assertRaises(ValueError):
            label_from_folder(Path("loitering/clip.mp4"))

    def test_holdout_is_fifteen_percent_of_each_class(self) -> None:
        self.assertEqual(split_sizes(90), (62, 14, 14))
        self.assertEqual(split_sizes(92), (64, 14, 14))

    def test_identical_files_stay_in_one_split_and_count_once(self) -> None:
        videos = []
        for index in range(6):
            videos.append(_video(f"normal/normal-{index}.mp4", "normal", f"hash-n-{index}"))
        videos.append(_video("normal/normal-copy.mp4", "normal", "hash-n-0"))
        for index in range(6):
            videos.append(_video(f"shoplifting/shoplifting-{index}.mp4", "shoplifting", f"hash-s-{index}"))
        split = assign_splits(videos)
        locations = {}
        for split_name, rows in split["splits"].items():
            for row in rows:
                if row["sha256"] in locations:
                    self.assertEqual(locations[row["sha256"]], split_name)
                else:
                    locations[row["sha256"]] = split_name
        copies = [row for rows in split["splits"].values() for row in rows if row["sha256"] == "hash-n-0"]
        self.assertEqual(len(copies), 2)
        self.assertEqual({row["used_for_training"] for row in copies}, {True, False})
        self.assertEqual(split["behaviour_labels"], None)

    def test_conflicting_folder_labels_are_excluded(self) -> None:
        videos = [
            _video("normal/normal-1.mp4", "normal", "same"),
            _video("shoplifting/shoplifting-1.mp4", "shoplifting", "same"),
            _video("normal/normal-2.mp4", "normal", "n2"),
            _video("normal/normal-3.mp4", "normal", "n3"),
            _video("normal/normal-4.mp4", "normal", "n4"),
            _video("shoplifting/shoplifting-2.mp4", "shoplifting", "s2"),
            _video("shoplifting/shoplifting-3.mp4", "shoplifting", "s3"),
            _video("shoplifting/shoplifting-4.mp4", "shoplifting", "s4"),
        ]
        split = assign_splits(videos)
        excluded = {item["video_id"] for item in split["excluded"]}
        self.assertEqual(excluded, {"normal/normal-1.mp4", "shoplifting/shoplifting-1.mp4"})


class RealVideoTests(unittest.TestCase):
    def test_catalog_matches_files_and_does_not_invent_behaviour_labels(self) -> None:
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(catalog["behaviour_labels"], None)
        self.assertEqual(split["behaviour_labels"], None)
        self.assertEqual(catalog["behaviour_types_not_annotated"], list(BEHAVIOUR_TYPES))
        self.assertEqual(catalog["missing_expected_names"]["shoplifting"], ["shoplifting-49.mp4"])
        self.assertEqual(catalog["missing_expected_names"]["normal"], [])
        self.assertEqual(catalog["counts"]["normal"], 90)
        self.assertEqual(catalog["counts"]["shoplifting"], 92)
        seen = set()
        for video in catalog["videos"]:
            self.assertIn(video["label"], {"normal", "shoplifting"})
            self.assertIsNone(video["behaviour_labels"])
            self.assertEqual(video["label_source"], "parent_folder")
            self.assertNotIn(video["video_id"], seen)
            seen.add(video["video_id"])
            path = VIDEO_ROOT / video["video_id"]
            if not path.is_file():
                self.skipTest(
                    f"Video dataset not available: {video['video_id']}"
                )
            self.assertEqual(path.stat().st_size, video["bytes"])
            self.assertEqual(file_sha256(path), video["sha256"])
            self.assertTrue(video["readable"])
            self.assertGreaterEqual(video["frames_decoded"], 2)
        rebuilt = assign_splits(catalog["videos"])
        self.assertEqual(rebuilt["splits"], split["splits"])
        ids = []
        for rows in split["splits"].values():
            ids.extend(row["video_id"] for row in rows)
        self.assertEqual(len(ids), len(set(ids)))

    def test_saved_test_metrics_match_a_fresh_fit_on_the_cached_features(self) -> None:
        features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
        committed = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        _model, fresh = train_from_features(features)
        self.assertFalse(committed["behaviour_labels_used"])
        self.assertEqual(committed["positive_class"], "shoplifting")
        self.assertEqual(fresh["test"]["confusion_matrix"], committed["test"]["confusion_matrix"])
        self.assertAlmostEqual(fresh["test"]["f1"], committed["test"]["f1"], places=6)
        self.assertAlmostEqual(fresh["validation"]["f1"], committed["validation"]["f1"], places=6)
        for behaviour in BEHAVIOUR_TYPES:
            self.assertNotIn(behaviour, committed)


def _video(video_id: str, label: str, digest: str) -> dict:
    return {
        "video_id": video_id,
        "label": label,
        "sha256": digest,
        "readable": True,
        "features": [0.0],
        "error": None,
    }


if __name__ == "__main__":
    unittest.main()
