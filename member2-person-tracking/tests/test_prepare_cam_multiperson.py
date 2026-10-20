import tempfile
import unittest
from pathlib import Path

from prepare_cam_multiperson import DATASET_YAML, prepare, split_images


class PrepareCamMultipersonTests(unittest.TestCase):
    def test_split_is_deterministic_and_holds_out_about_twenty_percent(self):
        images = [Path(f"cam_{index:02d}.jpg") for index in range(10)]
        first_train, first_val = split_images(images, seed=42)
        second_train, second_val = split_images(images, seed=42)
        self.assertEqual([path.name for path in first_train], [path.name for path in second_train])
        self.assertEqual([path.name for path in first_val], [path.name for path in second_val])
        self.assertEqual(len(first_train), 8)
        self.assertEqual(len(first_val), 2)
        self.assertEqual({path.name for path in first_train} & {path.name for path in first_val}, set())

    def test_missing_labels_become_empty_files_and_yaml_is_single_class(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "cam multi-person"
            output = root / "yolo_cam_multiperson"
            source.mkdir()
            for index in range(5):
                (source / f"frame_{index}.jpg").write_bytes(b"image")
            (source / "frame_0.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            summary = prepare(source, output, seed=42)
            self.assertEqual(summary["images"], 5)
            self.assertEqual(summary["train"] + summary["val"], 5)
            self.assertEqual(summary["labelled"], 1)
            self.assertEqual(summary["empty_labels"], 4)
            self.assertEqual((output / "dataset.yaml").read_text(encoding="utf-8"), DATASET_YAML)
            label_files = list((output / "labels").rglob("*.txt"))
            self.assertEqual(len(label_files), 5)
            self.assertEqual(sum(path.stat().st_size > 0 for path in label_files), 1)
            image_files = list((output / "images").rglob("*.jpg"))
            self.assertEqual(len(image_files), 5)


if __name__ == "__main__":
    unittest.main()
