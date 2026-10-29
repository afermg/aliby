"""Render fixed-target diagnostics without selecting ROIs by segmentation outcome."""

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries
from aliby_analyses.time_series_qc import PlaneReader

attempt = Path(sys.argv[1])
result = json.loads((attempt / "result.json").read_text())
job = json.loads((attempt / "attempt.json").read_text())
root = attempt.parents[1]
prepared = json.loads((root / "prepared" / job["site"] / "prepared.json").read_text())
output = Path(result["output"])
with np.load(output / "tiling" / job["site"] / "state.npz") as state:
    origins = state["crop_origins_yx"]
    centres = state["t0_centres_yx"]
targets = [(300, 300), (600, 600), (900, 900)]
selected = [
    int(np.argmin(((centres - np.asarray(t)) ** 2).sum(axis=1))) for t in targets
]
frames = sorted(set((0, result["ntps"] // 2, result["ntps"] - 1)))
canvas = Image.new("RGB", (1280, 420 * len(frames)), "white")
draw = ImageDraw.Draw(canvas)
reader = PlaneReader(Path(prepared["archive"]))
records = []
try:
    for row, tp in enumerate(frames):
        raw = reader.plane(tp, 2, 0)
        low, high = np.percentile(raw, (1, 99.5))
        grey = np.clip((raw - low) / (high - low) * 255, 0, 255).astype("uint8")
        overlay = np.repeat(grey[:, :, None], 3, axis=2)
        counts = []
        with np.load(
            output / "steps" / job["site"] / "segment_cell" / f"{tp:04d}.npz"
        ) as masks:
            for tile in range(len(centres)):
                layers = masks[f"tile_{tile}"]
                counts.append(int(np.count_nonzero(np.unique(layers) > 0)))
                y, x = origins[tp, tile]
                edge = np.zeros((117, 117), bool)
                for layer in layers:
                    edge |= find_boundaries(layer > 0, mode="inner")
                overlay[y : y + 117, x : x + 117][edge] = (255, 70, 50)
            canvas.paste(
                Image.fromarray(overlay).resize((400, 400)), (0, row * 420 + 20)
            )
            draw.text(
                (4, row * 420 + 3),
                f"{job['site']} tp{tp}: {sum(counts)} detections",
                fill="black",
            )
            for column, tile in enumerate(selected):
                y, x = origins[tp, tile]
                crop = raw[y : y + 117, x : x + 117]
                a, b = np.percentile(crop, (1, 99.5))
                cropgrey = np.clip((crop - a) / (b - a) * 255, 0, 255).astype("uint8")
                rgb = np.repeat(cropgrey[:, :, None], 3, axis=2)
                for layer in masks[f"tile_{tile}"]:
                    rgb[find_boundaries(layer > 0, mode="inner")] = (255, 70, 50)
                canvas.paste(
                    Image.fromarray(rgb).resize((280, 280)),
                    (410 + column * 290, row * 420 + 50),
                )
                draw.text(
                    (410 + column * 290, row * 420 + 20),
                    f"target {targets[column]} tile {tile}: {counts[tile]} detections",
                    fill="black",
                )
                draw.text(
                    (410 + column * 290, row * 420 + 340),
                    f"raw range {int(crop.min())}..{int(crop.max())}; local contrast",
                    fill="black",
                )
        records.append({"tp": tp, "detections": sum(counts), "per_tile": counts})
finally:
    reader.close()
canvas.save(attempt / "fixed-target-review-v2.png")
(attempt / "fixed-target-review-v2.json").write_text(
    json.dumps(
        {
            "targets_yx": targets,
            "tiles": selected,
            "frames": records,
            "human_reviewed": False,
            "contrast": "percentile for visualization, not calibration",
        },
        indent=2,
    )
    + "\n"
)
print(attempt / "fixed-target-review-v2.png")
