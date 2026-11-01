"""Read-only corpus census and channel-identity previews for the authorized complement."""

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

from aliby_analyses.channel_z import read_acquisition
from aliby_analyses.time_series_qc import PlaneReader

CACHE = Path("/work/users/amunoz/aliby_baby/cache")
PREVIOUS = Path(
    "/work/users/amunoz/aliby_baby/analyses/corpus-rebuild-20261021/batch-plan-v1.json"
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complement(manifest, previous):
    excluded = {j["record_key"] for j in previous["jobs"]} | {
        "14191444:gcd2_002",
        "14190058:gcd7_020",
    }
    candidates = []
    seen = set()
    for item in manifest:
        if not item["key"].endswith(".zip"):
            continue
        key = f"{item['deposition']}:{Path(item['key']).stem}"
        if key in seen:
            raise ValueError("Duplicate primary archive identity: " + key)
        seen.add(key)
        if key not in excluded:
            candidates.append(item)
    if len(candidates) != 282 or len(seen) != 388 or not excluded.issubset(seen):
        raise ValueError("Authorized complement no longer matches the 388-site census")
    return sorted(
        candidates,
        key=lambda x: (int(x["experiment"].split("_")[0]), x["deposition"], x["key"]),
    )


def build(root):
    rows = complement(
        json.loads((CACHE / "zenodo_manifest.json").read_text()),
        json.loads(PREVIOUS.read_text()),
    )
    proposals = []
    pages = []
    canvas = None
    for i, item in enumerate(rows):
        if i % 8 == 0:
            canvas = Image.new("RGB", (1250, 8 * 190), "white")
            pages.append(canvas)
        site = Path(item["key"]).stem
        key = f"{item['deposition']}:{site}"
        archive = CACHE / str(item["deposition"]) / item["key"]
        evidence = [read_acquisition(p, site) for p in archive.parent.glob("*Acq.txt")]
        matched = [e for e in evidence if e["status"] == "matched"]
        if len(matched) != 1:
            raise ValueError(
                "Needs explicit channel configuration, not a metadata-only rejection: "
                + key
            )
        meta = matched[0]
        channels = meta["site_channels"]
        with zipfile.ZipFile(archive) as zipped:
            headers = [n for n in zipped.namelist() if n.endswith("/.zarray")]
            if len(headers) != 1:
                raise ValueError("Ambiguous array: " + key)
            header = json.loads(zipped.read(headers[0]))
        if len(channels) != header["shape"][1] or header["shape"][2] != 5:
            raise ValueError("Channel/storage layout conflict: " + key)
        lines = Path(meta["path"]).read_text().splitlines()
        times = [
            int(float(v)) for v in lines[lines.index("Time_settings:") + 1].split(",")
        ]
        declared = {
            name: meta["channels"][name.lower()]["declared_planes"] for name in channels
        }
        model = f"yeast-alcatras-brightfield-sCMOS-60x-{declared['Brightfield']}z"
        record = {
            "record_id": item["deposition"],
            "experiment": item["experiment"],
            "part": int(item["part"]),
            "site": site,
            "archive": {k: item[k] for k in ("key", "size", "md5", "url")},
            "capture_order": "TCZYX",
            "shape": header["shape"],
            "channels": channels,
            "segmentation_channel": "Brightfield",
            "fluorescence_channels": [c for c in channels if c != "Brightfield"],
            "modelset": model,
            "tiling": {"tile_size": 117, "ref_z": 0},
            "observed_input": {
                "inner_zarr_path": headers[0].removesuffix("/.zarray"),
                "zarr_format": header["zarr_format"],
                "dtype": header["dtype"],
                "chunks": header["chunks"],
                "order": header["order"],
                "time_settings_raw": times,
            },
            "metadata_evidence": {
                "channels": f"Exact-site acquisition {meta['path']} SHA256={meta['sha256']}; proposed axis order pending pixel identity check; fluorescence names acquisition-derived, not spectral calibration.",
                "capture_order": f"ZIP {headers[0]}, explicit TCZYX; array T={header['shape'][0]}, declared T={times[2]}; conflicts preserved.",
                "modelset": f"Explicit operational {model}, matched acquired BF Z; not verified camera/pixel calibration; fresh computational gates required.",
                "tiling": "Prespecified 117 crop/refZ0; no per-outcome tuning; full-series geometric exclusion with original IDs required.",
            },
        }
        reader = PlaneReader(archive)
        metrics = []
        try:
            draw = ImageDraw.Draw(canvas)
            y = (i % 8) * 190
            draw.text(
                (5, y + 2),
                f"{i + 1}. {key} exp{item['experiment'].split('_')[0]} / proposed BF{channels.index('Brightfield')}",
                fill="black",
            )
            for c, name in enumerate(channels):
                # For identity only: observed planes, no generated missing pixels.
                frame = np.max(
                    [
                        reader.plane(min(30, header["shape"][0] - 1), c, z)
                        for z in (0, 2, 4)
                    ],
                    axis=0,
                )
                crop = frame[344:856, 344:856]
                lo, hi = np.percentile(crop, (1, 99.5))
                view = (
                    np.zeros(crop.shape, dtype="uint8")
                    if hi == lo
                    else np.clip((crop - lo) / (hi - lo) * 255, 0, 255).astype("uint8")
                )
                canvas.paste(
                    Image.fromarray(view).resize((170, 150)), (430 + c * 200, y + 28)
                )
                draw.text((430 + c * 200, y + 12), f"C{c} {name}", fill="black")
                metrics.append(
                    {
                        "channel": c,
                        "min": int(frame.min()),
                        "median": float(np.median(frame)),
                        "max": int(frame.max()),
                    }
                )
        finally:
            reader.close()
        proposals.append(
            {
                "key": key,
                "archive": str(archive),
                "record": record,
                "declared_planes_by_name": declared,
                "acquisition": meta,
                "preview_page": f"review/channels-{i // 8 + 1:02d}.jpg",
                "preview_row": i % 8 + 1,
                "pixel_metrics": metrics,
            }
        )
        print(key, flush=True)
    for i, page in enumerate(pages):
        page.save(root / f"review/channels-{i + 1:02d}.jpg", quality=85)
    with (root / "proposals.json").open("x") as handle:
        json.dump(
            {
                "schema": "aliby.corpus-expansion-proposals.v1",
                "authorization": "User explicitly approved the remaining 282 positions, computational gates first; biological review deferred.",
                "source_manifest_sha256": sha(CACHE / "zenodo_manifest.json"),
                "previous_plan_sha256": sha(PREVIOUS),
                "sites": proposals,
            },
            handle,
            indent=2,
        )
        handle.write("\n")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
