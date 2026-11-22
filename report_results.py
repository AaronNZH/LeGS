import json
import math
import os
import sys
from pathlib import Path


SCENE_NAMES = {
    "mipnerf360": ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"],
    "tanks_and_temples": ["truck", "train"],
    "deep_blending": ["drjohnson", "playroom"],
}


def read_json(path):
    with path.open() as file:
        return json.load(file)


def finite_number(value, field):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return value


def nonnegative_number(value, field):
    value = finite_number(value, field)
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def mean(values):
    total = 0.0
    for value in values:
        total += value
    return total / len(values)


def read_scene(path):
    metrics = read_json(path / "results.json")["ours_30000"]
    statistics = read_json(path / "training_statistics.json")
    count = nonnegative_number(statistics["GS_number"], "GS_number")
    if count != int(count):
        raise ValueError("GS_number must be an integer")
    return (
        finite_number(metrics["SSIM"], "SSIM"),
        finite_number(metrics["PSNR"], "PSNR"),
        finite_number(metrics["LPIPS"], "LPIPS"),
        int(count),
        nonnegative_number(statistics["total"]["time_s"], "total.time_s"),
        nonnegative_number(statistics["peak_gpu_memory_mb"], "peak_gpu_memory_mb"),
    )


def main():
    output_path = Path(os.environ.get("LEGS_OUTPUT_ROOT") or "output/rl")
    datasets = {}
    errors = []
    for dataset, scene_names in SCENE_NAMES.items():
        rows = []
        scene_errors = []
        for scene in scene_names:
            path = output_path / scene
            try:
                rows.append(read_scene(path))
            except (OSError, ValueError, OverflowError, KeyError, TypeError) as exc:
                scene_errors.append(f"  {scene} ({path}): {exc}")
        datasets[dataset] = rows
        if scene_errors:
            errors.append(f"{dataset}: {len(rows)}/{len(scene_names)} valid scenes")
            errors.extend(scene_errors)

    if errors:
        print("Cannot report a complete suite:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1

    for dataset, rows in datasets.items():
        print(f"{dataset} ({len(rows)}/{len(SCENE_NAMES[dataset])} scenes)")
        ssims, psnrs, lpips, counts, times, peaks = zip(*rows)
        print(f"SSIM: {mean(ssims):.4f}, PSNR: {mean(psnrs):.4f}, "
              f"LPIP: {mean(lpips):.4f}, NGS: {mean(counts) / 1e6:.4f}, "
              f"Training time: {mean(times) / 60.:.4f} m, Peak Memory: {max(peaks):.4f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
