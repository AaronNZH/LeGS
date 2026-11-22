import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "report_results.py"
SCENES = (
    "bicycle", "flowers", "garden", "stump", "treehill", "room", "counter",
    "kitchen", "bonsai", "truck", "train", "drjohnson", "playroom",
)


class ReportResultsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_scene(self, name, psnr=25.74, step=30000, stats=True,
                    peak=2048.0, path="output/rl"):
        scene_dir = self.root / path / name
        scene_dir.mkdir(parents=True)
        (scene_dir / "results.json").write_text(json.dumps({
            f"ours_{step}": {"SSIM": 0.8, "PSNR": psnr, "LPIPS": 0.2},
        }))
        if stats:
            (scene_dir / "training_statistics.json").write_text(json.dumps({
                "GS_number": 1_000_000,
                "total": {"time_s": 60.0},
                "peak_gpu_memory_mb": peak,
            }))

    def run_report(self, output_root=None):
        env = os.environ.copy()
        env.pop("LEGS_OUTPUT_ROOT", None)
        if output_root is not None:
            env["LEGS_OUTPUT_ROOT"] = output_root
        return subprocess.run([sys.executable, str(SCRIPT)], cwd=str(self.root),
                              env=env, capture_output=True, text=True)

    def assert_incomplete(self, result):
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Cannot report a complete suite:", result.stderr)
        self.assertNotIn("PSNR:", result.stdout)

    def test_complete_preserves_scene_means_and_peak_max(self):
        for name in SCENES:
            self.write_scene(name, psnr=27.09 if name == "bicycle" else 25.74,
                             peak=4096 if name == "bicycle" else 2048)
        result = self.run_report()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mipnerf360 (9/9 scenes)", result.stdout)
        self.assertIn("PSNR: 25.8900", result.stdout)
        self.assertEqual(result.stdout.count("PSNR: 25.7400"), 2)
        self.assertIn("NGS: 1.0000, Training time: 1.0000 m, Peak Memory: 4096.0000 MB",
                      result.stdout)

    def test_missing(self):
        self.write_scene("bicycle")
        result = self.run_report()
        self.assert_incomplete(result)
        self.assertIn("mipnerf360: 1/9 valid scenes", result.stderr)
        self.assertIn("flowers", result.stderr)

    def test_empty(self):
        result = self.run_report()
        self.assert_incomplete(result)
        self.assertIn("mipnerf360: 0/9 valid scenes", result.stderr)
        self.assertIn("deep_blending: 0/2 valid scenes", result.stderr)

    def test_wrong_step(self):
        self.write_scene("bicycle", step=7000)
        result = self.run_report()
        self.assert_incomplete(result)
        self.assertIn("ours_30000", result.stderr)

    def test_missing_stats(self):
        self.write_scene("bicycle", stats=False)
        result = self.run_report()
        self.assert_incomplete(result)
        self.assertIn("output/rl/bicycle/training_statistics.json", result.stderr)

    def test_invalid_and_nonfinite_metrics(self):
        for i, invalid in enumerate((float("nan"), float("inf"), "25.74", True)):
            with self.subTest(invalid=invalid):
                path = f"output/invalid {i}"
                self.write_scene("bicycle", psnr=invalid, path=path)
                result = self.run_report(path)
                self.assert_incomplete(result)
                self.assertIn("PSNR must be a finite number", result.stderr)

    def test_invalid_stats(self):
        self.write_scene("bicycle")
        p = self.root / "output/rl/bicycle/training_statistics.json"
        data = json.loads(p.read_text())
        data["GS_number"] = 1.5
        p.write_text(json.dumps(data))
        result = self.run_report()
        self.assert_incomplete(result)
        self.assertIn("GS_number must be an integer", result.stderr)

    def test_invalid_json(self):
        self.write_scene("bicycle")
        (self.root / "output/rl/bicycle/results.json").write_text("{")
        result = self.run_report()
        self.assert_incomplete(result)
        self.assertIn("bicycle", result.stderr)
        self.assertIn("Expecting property name", result.stderr)

    def test_custom_output_root(self):
        path = "other runs/run 2"
        for name in SCENES:
            self.write_scene(name, path=path)
        result = self.run_report(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("deep_blending (2/2 scenes)", result.stdout)


if __name__ == "__main__":
    unittest.main()
