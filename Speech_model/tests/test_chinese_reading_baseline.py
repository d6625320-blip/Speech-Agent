"""End-to-end regression checks for the read-only diagnostic evaluator."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate_chinese_reading_baseline as baseline
import chinese_reading_joint_utils as utils
from mcae_large_final_common import LABELS
from test_chinese_reading_joint import reading_rows


class BaselineDiagnosticsTests(unittest.TestCase):
    def test_reports_both_heads_by_split_without_touching_test_or_checkpoints(self):
        torch.set_num_threads(1)
        temp_parent = Path(__file__).resolve().parents[2] / ".tmp/baseline_diagnostic_tests"
        temp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            rows, config = reading_rows()
            model_dir = root / "encoder"
            model_dir.mkdir()
            (model_dir / "model.pt").write_bytes(b"test encoder fingerprint, never loaded")
            fingerprint = utils.encoder_signature(model_dir)
            features = []
            for index, row in enumerate(rows):
                audio, feature = root / f"audio-{index}.wav", root / f"feature-{index}.npy"
                audio.write_bytes(f"audio-{index}".encode())
                row["audio_path"] = str(audio)
                np.save(feature, np.eye(7, dtype=np.float32)[int(row["label_id"])])
                features.append({**row, "feature_path": str(feature), "feature_dim": "7",
                                 "encoder_sha256": fingerprint, "audio_sha256": utils.sha256(audio),
                                 "feature_sha256": utils.sha256(feature)})
            for path, data in ((root / "source.csv", rows), (root / "features.csv", features)):
                baseline.write_csv(path, data)
            config["paths"] = {"model_dir": str(model_dir), "reading_manifest": str(root / "source.csv"),
                               "reading_feature_manifest": str(root / "features.csv"),
                               "output_dir": str(root / "joint")}
            config_file = root / "config.yaml"
            config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
            checkpoint = {"input_dim": 7, "hidden_dim": 0, "dropout": 0, "labels": LABELS,
                          "embedding_mean": np.zeros(7, dtype=np.float32),
                          "embedding_std": np.ones(7, dtype=np.float32), "temperature": 9.0,
                          "model_state": {"net.weight": torch.eye(7) * 5, "net.bias": torch.zeros(7)}}
            original_path = root / "original.pt"
            torch.save(checkpoint, original_path)
            joint_path = root / "joint/final_model.pt"
            joint_path.parent.mkdir()
            checkpoint["model_state"]["net.weight"][1, :] = 0
            checkpoint["model_state"]["net.weight"][6, 1] = 5
            checkpoint.update({"pipeline_version": utils.PIPELINE_VERSION,
                               "reading_manifest_identity": utils.manifest_identity(features),
                               "encoder_sha256": fingerprint})
            torch.save(checkpoint, joint_path)
            original_hash, joint_hash = utils.sha256(original_path), utils.sha256(joint_path)
            base_args = ["evaluate", "--config", str(config_file), "--original-checkpoint", str(original_path)]
            # Default diagnostics must not access even a single held-out test feature.
            heldout = [(Path(r["feature_path"]), Path(r["feature_path"]).read_bytes())
                       for r in features if r["split"] == "test"]
            for path, _ in heldout:
                path.unlink()
            output = root / "diagnostics"
            with patch.object(sys, "argv", base_args + ["--output-dir", str(output)]):
                baseline.main()
            report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["samples"], 960)
            self.assertEqual(set(report["per_split"]), {"train", "validation"})
            self.assertEqual(report["per_split"]["validation"]["original"]["accuracy"], 1)
            self.assertAlmostEqual(report["per_split"]["validation"]["joint"]["accuracy"], 5 / 6)
            original_probs = baseline.read_csv(output / "predictions.csv")
            self.assertTrue(all(float(r[f"original_prob_{r['label']}"]) > .9 for r in original_probs))
            self.assertEqual(len(baseline.read_csv(output / "comparison.csv")), 4)
            train_report = baseline.read_csv(output / "train_joint_class_metrics.csv")
            self.assertEqual(next(r for r in train_report if r["label"] == "disgust")["recall"], "0.0")
            matrix = baseline.read_csv(output / "validation_joint_confusion_counts.csv")
            self.assertEqual(next(r for r in matrix if r["true_label"] == "disgust")["surprised"], "40")
            with patch.object(sys, "argv", base_args + ["--output-dir", str(output)]):
                with self.assertRaises(FileExistsError):
                    baseline.main()
            for path, data in heldout:
                path.write_bytes(data)
            for split, count in (("test", 240), ("all", 1200)):
                destination = root / split
                with patch.object(sys, "argv", base_args + ["--split", split, "--output-dir", str(destination)]):
                    baseline.main()
                value = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(value["samples"], count)
                self.assertEqual(value["per_split"]["test"]["original"]["samples"], 240)
            joint_path.unlink()
            with patch.object(sys, "argv", base_args + ["--split", "validation", "--original-only",
                                                       "--output-dir", str(root / "original_only")]):
                baseline.main()
            original_only = json.loads((root / "original_only/metrics.json").read_text())
            self.assertNotIn("joint", original_only)
            self.assertEqual(utils.sha256(original_path), original_hash)
            # Checkpoint hash in the first report must match the exact joint weights used.
            self.assertEqual(report["joint_checkpoint_sha256"], joint_hash)


if __name__ == "__main__":
    unittest.main()
