"""Regression tests for sampling, split isolation, provenance and result reset."""

import contextlib
import copy
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
import chinese_reading_joint_utils as utils
import train_mcae_chinese_reading_joint as train
import evaluate_mcae_chinese_reading_joint as evaluate
import extract_chinese_reading_embeddings as extract
from mcae_large_final_common import LABELS, load_final_head


def reading_rows():
    splits = {
        "train": ["SPK0001", "SPK0002", "SPK0003", "SPK0006", "SPK0007", "SPK0008"],
        "validation": ["SPK0004", "SPK0009"], "test": ["SPK0005", "SPK0010"],
    }
    rows = []
    for split, speakers in splits.items():
        for speaker in speakers:
            for index in range(120):
                label = utils.READING_LABELS[index // 20]
                rows.append({"sample_id": f"{speaker}_{index}", "speaker_id": speaker,
                             "split": split, "label": label, "label_id": str(LABELS.index(label)),
                             "prompt_id": str(index), "duration": "4.5"})
    return rows, {"splits": splits}


class JointRegressionTests(unittest.TestCase):
    def test_sampling_probability_and_actual_draws(self):
        for source_mass in (.75, .5):
            weights = utils.build_sampling_weights(6675, 720, source_mass, 1 - source_mass)
            self.assertAlmostEqual(weights[:6675].sum(), source_mass)
            self.assertAlmostEqual(weights[6675:].sum(), 1 - source_mass)
            draws = list(torch.utils.data.WeightedRandomSampler(
                torch.tensor(weights), 90000, replacement=True,
                generator=torch.Generator().manual_seed(42)))
            fraction = sum(i >= 6675 for i in draws) / len(draws)
            self.assertAlmostEqual(fraction, 1 - source_mass, delta=.01)

    def test_invalid_sampling_mass_rejected(self):
        for masses in ((float("nan"), .25), (.75, -.25), (.8, .8)):
            with self.assertRaises(ValueError):
                utils.build_sampling_weights(6675, 720, *masses)

    def test_stale_splits_and_duplicate_ids_rejected(self):
        rows, config = reading_rows()
        utils.validate_reading_rows(rows, config)
        changed = copy.deepcopy(config)
        changed["splits"]["validation"], changed["splits"]["test"] = (
            changed["splits"]["test"], changed["splits"]["validation"])
        with self.assertRaisesRegex(ValueError, "disagrees"):
            utils.validate_reading_rows(rows, changed)
        bad = copy.deepcopy(rows)
        bad[-1]["sample_id"] = bad[0]["sample_id"]
        with self.assertRaises(ValueError):
            utils.validate_reading_rows(bad, config)

    def test_stale_feature_labels_rejected(self):
        rows, _ = reading_rows()
        changed = copy.deepcopy(rows)
        changed[0]["label"] = "surprised"
        with self.assertRaises(ValueError):
            utils.require_matching_manifests(rows, changed)

    def test_cache_detects_audio_feature_and_encoder_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio, feature = root / "a.wav", root / "a.npy"
            audio.write_bytes(b"audio-version-one")
            np.save(feature, np.ones(7, dtype=np.float32))
            source = [{"sample_id": "a", "audio_path": str(audio)}]
            row = {"sample_id": "a", "feature_path": str(feature), "encoder_sha256": "encoder-one",
                   "audio_sha256": utils.sha256(audio), "feature_sha256": utils.sha256(feature)}
            utils.verify_reading_cache([row], "encoder-one", source)
            with self.assertRaises(ValueError):
                utils.verify_reading_cache([row], "encoder-two", source)
            audio.write_bytes(b"audio-version-two")
            with self.assertRaisesRegex(ValueError, "Audio changed"):
                utils.verify_reading_cache([row], "encoder-one", source)
            audio.write_bytes(b"audio-version-one")
            np.save(feature, np.zeros(7, dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "corrupt"):
                utils.verify_reading_cache([row], "encoder-one", source)

    def test_old_windows_paths_relocate_to_current_project(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            feature = root / "features/emotion2vec_plus_large/actor_01/a.npy"
            feature.parent.mkdir(parents=True)
            np.save(feature, np.ones(7))
            with patch.object(utils, "PROJECT_DIR", root):
                self.assertEqual(utils.resolve_feature_path(
                    r"F:\example\Speech_model\features\emotion2vec_plus_large\actor_01\a.npy"), feature)

    def test_reset_only_removes_authorized_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protected = root / "outputs/final/mcae_large_class_balanced_ce/final_model.pt"
            protected.parent.mkdir(parents=True)
            protected.write_bytes(b"keep original MCAE model")
            targets = [root / "outputs/final/mcae_chinese_reading_joint_v1",
                       root / "outputs/evaluation/mcae_chinese_reading_joint_v1"]
            for target in targets:
                target.mkdir(parents=True)
                (target / "old-result.txt").write_text("discard")
            config = {"paths": {"output_dir": str(targets[0]), "test_output_dir": str(targets[1])}}
            with patch.object(utils, "PROJECT_DIR", root):
                bad = copy.deepcopy(config)
                bad["paths"]["output_dir"] = str(protected.parent)
                with self.assertRaises(ValueError):
                    utils.reset_results(bad)
                self.assertTrue(targets[0].exists())
                utils.reset_results(config)
            self.assertEqual(protected.read_bytes(), b"keep original MCAE model")
            self.assertTrue(all(not target.exists() for target in targets))

    def test_extraction_rebuilds_legacy_cache_then_reuses_verified_cache(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            rows, config = reading_rows()
            model_dir = root / "encoder"
            model_dir.mkdir()
            (model_dir / "model.pt").write_bytes(b"fake encoder for protocol test")
            for index, row in enumerate(rows):
                audio = root / f"audio-{index}.wav"
                audio.write_bytes(f"audio {index}".encode())
                row["audio_path"] = str(audio)
            config.update({"paths": {"reading_manifest": str(root / "source.csv"),
                                      "reading_feature_dir": str(root / "cache"),
                                      "reading_feature_manifest": str(root / "features.csv"),
                                      "model_dir": str(model_dir)},
                           "extraction": {"device": "cpu", "batch_size": 64, "overwrite": False}})
            train.write_csv(root / "source.csv", rows)
            config_file = root / "config.yaml"
            config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
            first = rows[0]
            legacy = root / "cache" / first["split"] / first["speaker_id"] / first["label"] / (first["sample_id"] + ".npy")
            legacy.parent.mkdir(parents=True)
            np.save(legacy, np.full(7, -999, dtype=np.float32))
            extracted = []

            class FakeExtractor:
                def __init__(self, *args, **kwargs):
                    pass

                def extract_many(self, paths, batch_size):
                    extracted.extend(paths)
                    return np.ones((len(paths), 7), dtype=np.float32)

            with patch.object(extract, "Emotion2VecEmbeddingExtractor", FakeExtractor), \
                    patch.object(sys, "argv", ["extract", "--config", str(config_file)]):
                extract.main()
            self.assertEqual(len(extracted), 1200)
            np.testing.assert_array_equal(np.load(legacy), np.ones(7))
            with patch.object(extract, "Emotion2VecEmbeddingExtractor", side_effect=AssertionError("cache should be reused")), \
                    patch.object(sys, "argv", ["extract", "--config", str(config_file)]):
                extract.main()
            Path(rows[0]["audio_path"]).write_bytes(b"changed audio")
            extracted.clear()
            with patch.object(extract, "Emotion2VecEmbeddingExtractor", FakeExtractor), \
                    patch.object(sys, "argv", ["extract", "--config", str(config_file)]):
                extract.main()
            self.assertEqual(extracted, [Path(rows[0]["audio_path"])])

    def test_complete_training_and_evaluation_with_synthetic_cached_features(self):
        # Executes the real CLI entry points and checkpoint reload without a GPU/encoder inference.
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            rows, config = reading_rows()
            model_dir = root / "encoder"
            model_dir.mkdir()
            (model_dir / "model.pt").write_bytes(b"synthetic encoder identity")
            fingerprint = utils.encoder_signature(model_dir)
            mcae = []
            for index in range(6675):
                label_id = index % 7
                path = root / f"mcae-{label_id}.npy"
                if not path.exists():
                    np.save(path, np.eye(7, dtype=np.float32)[label_id])
                mcae.append({"sample_id": str(index), "label_id": str(label_id),
                             "label": LABELS[label_id], "feature_path": str(path)})
            features = []
            for index, row in enumerate(rows):
                audio, feature = root / f"a{index}.wav", root / f"f{index}.npy"
                audio.write_bytes(f"synthetic audio {index}".encode())
                row["audio_path"] = str(audio)
                np.save(feature, np.eye(7, dtype=np.float32)[int(row["label_id"])])
                features.append({**row, "feature_path": str(feature), "feature_dim": "7",
                                 "encoder_sha256": fingerprint, "audio_sha256": utils.sha256(audio),
                                 "feature_sha256": utils.sha256(feature)})
            for name, data in (("mcae.csv", mcae), ("source.csv", rows), ("features.csv", features)):
                train.write_csv(root / name, data)
            config.update({
                "experiment": {"seed": 42}, "model": {"embedding_dim": 7, "hidden_dim": 0, "dropout": 0},
                "training": {"device": "cpu", "epochs": 2, "batch_size": 64, "num_workers": 0,
                             "head_learning_rate": .01, "weight_decay": .001, "class_weight_beta": .999,
                             "mcae_mass": .75, "reading_mass": .25, "samples_per_epoch": 9000,
                             "early_stopping_patience": 2},
                "calibration": {"enabled": False, "ece_bins": 15},
                "paths": {"mcae_feature_manifest": str(root / "mcae.csv"),
                          "reading_manifest": str(root / "source.csv"),
                          "reading_feature_manifest": str(root / "features.csv"),
                          "model_dir": str(model_dir), "output_dir": str(root / "trained"),
                          "test_output_dir": str(root / "evaluated")},
            })
            config_file = root / "config.yaml"
            config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
            # Training must not load test features: make one unreadable until training finishes.
            heldout = next(r for r in features if r["split"] == "test")
            heldout_path = Path(heldout["feature_path"])
            heldout_bytes = heldout_path.read_bytes()
            heldout_path.unlink()
            with patch.object(sys, "argv", ["train", "--config", str(config_file)]):
                train.main()
            heldout_path.write_bytes(heldout_bytes)
            _, checkpoint, _, _, temperature = load_final_head(root / "trained/final_model.pt", torch.device("cpu"))
            self.assertEqual(checkpoint["pipeline_version"], utils.PIPELINE_VERSION)
            self.assertEqual(temperature, 1)
            history = json.loads((root / "trained/sampling_history.json").read_text())
            self.assertTrue(all(abs(r["reading_fraction"] - .25) < .02 for r in history))
            self.assertTrue((root / "trained/validation_metrics.json").is_file())
            with patch.object(sys, "argv", ["evaluate", "--config", str(config_file)]):
                evaluate.main()
            result = json.loads((root / "evaluated/metrics.json").read_text())
            self.assertEqual(result["samples"], 240)
            self.assertEqual(result["accuracy"], 1.0)
            with patch.object(sys, "argv", ["evaluate", "--config", str(config_file)]):
                with self.assertRaises(FileExistsError):
                    evaluate.main()


if __name__ == "__main__":
    unittest.main()
